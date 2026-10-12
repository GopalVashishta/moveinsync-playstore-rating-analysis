#!/usr/bin/env python3
"""Preprocess parsed Google Play reviews from multiple text files.

Expected input: each entry is formatted as:
    Reviewer / developer display name
    August 26, 2019
    Entry text...

A developer reply is recognized using DEVELOPER_ALIASES below and attached to
only the immediately preceding user review. The script does not use NLP models
or make network calls.

Outputs (created in ./processed_reviews beside this script):
    reviews_preprocessed.csv  Deduplicated structured reviews
    reviews_ranked.csv       Same reviews sorted by priority_score
    parsing_issues.csv       Orphan developer replies / unparsed prefixes
    processing_summary.txt   Counts and configuration summary
    # Removed "Did you find this helpful?" part which came as noise in gathering reviews
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration: edit these values if needed
# ---------------------------------------------------------------------------

INPUT_FILES = [
    "reviews/most_relevant.txt",
    "reviews/newest_1start.txt",
    "reviews/newest_2star_apr_12oct.txt",
    "reviews/newest_3star_apr_oct.txt",
    "reviews/newest_4star_jun_oct.txt",
]

# A block whose author contains any alias (case-insensitive) is considered a
# company/developer response rather than a user review.
DEVELOPER_ALIASES = [
    "MoveInSync",
    # Add other displayed developer names / aliases here, if applicable.
]

OUTPUT_DIR_NAME = "processed_reviews"
RECENCY_HALF_LIFE_DAYS = 730  # time weight halves every two years
RECENCY_WEIGHT = 0.70
HELPFULNESS_WEIGHT = 0.30

# ---------------------------------------------------------------------------
# Text/date patterns
# ---------------------------------------------------------------------------

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

DATE_PATTERNS = [
    re.compile(rf"^(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}$", re.IGNORECASE),
    re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"),
    re.compile(r"^\d{1,2}\s+(?:" + MONTHS + r")\s+\d{4}$", re.IGNORECASE),
]

HELPFUL_COUNT_RE = re.compile(
    r"^(\d+)\s+(?:people|person)\s+found\s+"
    r"(?:this(?:\s+review)?\s+)?(?:helpful|useful)[.!]?$",
    re.IGNORECASE,
)

HELPFUL_QUESTION_RE = re.compile(
    r"^did\s+(?:you|u)\s+find\s+this(?:\s+review)?\s+"
    r"(?:helpful|useful)\s*\??[.!]?$",
    re.IGNORECASE,
)

# Example filenames: newest_1start.txt, newest_2star_apr_12oct.txt
RATING_FROM_FILENAME_RE = re.compile(r"(?:^|[_-])([1-5])\s*star", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving the original words and punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def parse_date_line(value: str) -> date | None:
    """Parse common Google Play date-line formats, otherwise return None."""
    value = value.strip()
    if not value or not any(pattern.fullmatch(value) for pattern in DATE_PATTERNS):
        return None

    formats = (
        "%B %d, %Y", "%b %d, %Y", "%B %d, %Y",
        "%Y-%m-%d", "%Y-%m-%d", "%d %B %Y", "%d %b %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value.title(), fmt).date()
        except ValueError:
            continue
    return None


def helpful_count_from_line(line: str) -> int | None:
    match = HELPFUL_COUNT_RE.fullmatch(line.strip())
    return int(match.group(1)) if match else None


def clean_review_body(body: str) -> tuple[str, int | None]:
    """Remove helpful-vote UI lines and the 'Did you find this useful?' prompt."""
    helpful_counts: list[int] = []
    retained_lines: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        count = helpful_count_from_line(line)
        if count is not None:
            helpful_counts.append(count)
            continue

        if HELPFUL_QUESTION_RE.fullmatch(line):
            continue

        retained_lines.append(line)

    # In rare cases there may be more than one count-like line. Preserve the
    # largest observed count within this entry and flagging isn't needed for a
    # normal Google Play block.
    helpful_count = max(helpful_counts) if helpful_counts else None
    return normalize_whitespace(" ".join(retained_lines)), helpful_count


def infer_rating_from_filename(filename: str) -> int | None:
    match = RATING_FROM_FILENAME_RE.search(Path(filename).stem)
    return int(match.group(1)) if match else None


def is_developer_author(author: str, aliases: list[str]) -> bool:
    folded = author.casefold()
    return any(alias.casefold() in folded for alias in aliases if alias.strip())


def read_text_robustly(path: Path) -> str:
    """Read UTF-8 first, then use Windows-1252 for legacy text exports."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252")


def find_entry_headers(lines: list[str]) -> list[dict[str, Any]]:
    """Find entries marked by an author line immediately before a date line."""
    headers: list[dict[str, Any]] = []

    for date_idx, raw_line in enumerate(lines):
        entry_date = parse_date_line(raw_line)
        if entry_date is None:
            continue

        author_idx = date_idx - 1
        while author_idx >= 0 and not lines[author_idx].strip():
            author_idx -= 1

        if author_idx < 0:
            continue

        author = lines[author_idx].strip()
        # Avoid interpreting obvious UI prompts as author headers if a malformed
        # input happens to put a date immediately after one.
        if HELPFUL_QUESTION_RE.fullmatch(author):
            continue

        headers.append({
            "author_idx": author_idx,
            "date_idx": date_idx,
            "author": author,
            "date": entry_date,
        })

    return headers


def parse_one_file(
    path: Path,
    aliases: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Parse one input file and attach replies in sequence order."""
    raw_text = read_text_robustly(path)
    lines = raw_text.replace("\ufeff", "").splitlines()
    headers = find_entry_headers(lines)

    if not headers:
        raise ValueError(
            f"No review/date entries found in {path.name}. Expected entries "
            "with an author line followed by a date such as 'August 26, 2019'."
        )

    issues: list[dict[str, Any]] = []
    first_header_idx = headers[0]["author_idx"]
    unmatched_prefix = "\n".join(lines[:first_header_idx]).strip()
    if unmatched_prefix:
        issues.append({
            "source_file": path.name,
            "issue_type": "unparsed_prefix_before_first_entry",
            "author": None,
            "entry_date": None,
            "text": unmatched_prefix,
        })

    # Entry body extends until the author line of the next date-marked entry.
    blocks: list[dict[str, Any]] = []
    for i, header in enumerate(headers):
        end_idx = headers[i + 1]["author_idx"] if i + 1 < len(headers) else len(lines)
        body = "\n".join(lines[header["date_idx"] + 1:end_idx]).strip()
        blocks.append({
            "author": header["author"],
            "entry_date": header["date"],
            "body": body,
        })

    file_rating = infer_rating_from_filename(path.name)
    records: list[dict[str, Any]] = []
    orphan_reply_count = 0

    for block in blocks:
        author = block["author"]
        body = block["body"]
        entry_date = block["entry_date"]

        if is_developer_author(author, aliases):
            # Do not attach a reply to an older review across intervening
            # entries. It must follow the review immediately in the source.
            target = records[-1] if records else None
            if target is not None and target["developer_reply_text"] is None:
                target["developer_reply_author"] = author
                target["developer_reply_date"] = entry_date
                target["developer_reply_text"] = normalize_whitespace(body) or None
            else:
                orphan_reply_count += 1
                issues.append({
                    "source_file": path.name,
                    "issue_type": "orphan_developer_reply",
                    "author": author,
                    "entry_date": entry_date.isoformat(),
                    "text": normalize_whitespace(body),
                })
            continue

        review_text, helpful_count = clean_review_body(body)
        raw_review_block = body
        records.append({
            "reviewer_name": author,
            "review_date": entry_date,
            "review_text_clean": review_text,
            "helpful_count": helpful_count,
            "developer_reply_author": None,
            "developer_reply_date": None,
            "developer_reply_text": None,
            "has_developer_reply": False,
            "raw_review_block": raw_review_block,
            "source_files": [path.name],
            "source_ratings": [file_rating] if file_rating is not None else [],
            "helpful_counts_observed": [helpful_count] if helpful_count is not None else [],
            "source_occurrences": 1,
        })

    stats = {
        "entries_detected": len(blocks),
        "user_reviews_parsed": len(records),
        "developer_blocks_detected": sum(1 for b in blocks if is_developer_author(b["author"], aliases)),
        "orphan_developer_replies": orphan_reply_count,
        "reviews_with_helpful_count": sum(r["helpful_count"] is not None for r in records),
        "reviews_with_reply": sum(r["developer_reply_text"] is not None for r in records),
    }
    return records, issues, stats


def canonical_text(value: Any) -> str:
    """Canonical representation used only for exact/near-exact deduplication."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    return normalize_whitespace(value).casefold()


def make_fingerprint(record: dict[str, Any]) -> str:
    key = "\x1f".join([
        canonical_text(record.get("reviewer_name")),
        pd.Timestamp(record["review_date"]).strftime("%Y-%m-%d"),
        canonical_text(record.get("review_text_clean")),
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def merge_duplicate_reviews(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge identical reviews appearing in more than one input file."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[make_fingerprint(record)].append(record)

    merged_records: list[dict[str, Any]] = []
    for fingerprint, group in groups.items():
        merged = dict(group[0])

        all_sources: list[str] = []
        all_ratings: list[int] = []
        all_votes: list[int] = []
        reply_candidates: list[dict[str, Any]] = []

        for item in group:
            for source in item.get("source_files", []):
                if source not in all_sources:
                    all_sources.append(source)
            for rating in item.get("source_ratings", []):
                if rating not in all_ratings:
                    all_ratings.append(rating)
            for count in item.get("helpful_counts_observed", []):
                if count not in all_votes:
                    all_votes.append(count)
            if item.get("developer_reply_text"):
                reply_candidates.append(item)

        merged["review_id"] = f"R_{fingerprint}"
        merged["review_fingerprint"] = fingerprint
        merged["source_files"] = "; ".join(all_sources)
        merged["source_count"] = len(all_sources)
        merged["source_ratings"] = "; ".join(str(r) for r in sorted(all_ratings))
        merged["rating_conflict"] = len(all_ratings) > 1
        merged["rating"] = all_ratings[0] if len(all_ratings) == 1 else pd.NA
        merged["helpful_counts_observed"] = "; ".join(
            str(v) for v in sorted(all_votes)
        )
        # Counts are snapshots and may differ across files; use the maximum
        # observed count for ranking, while preserving all values above.
        merged["helpful_count"] = max(all_votes) if all_votes else pd.NA
        merged["source_occurrences"] = sum(
            int(item.get("source_occurrences", 1)) for item in group
        )

        if reply_candidates:
            # Choose the most recent dated reply if duplicates disagree.
            reply_candidates.sort(
                key=lambda item: item.get("developer_reply_date") or date.min
            )
            selected_reply = reply_candidates[-1]
            merged["developer_reply_author"] = selected_reply.get("developer_reply_author")
            merged["developer_reply_date"] = selected_reply.get("developer_reply_date")
            merged["developer_reply_text"] = selected_reply.get("developer_reply_text")
        else:
            merged["developer_reply_author"] = None
            merged["developer_reply_date"] = None
            merged["developer_reply_text"] = None

        merged["has_developer_reply"] = bool(merged.get("developer_reply_text"))
        merged_records.append(merged)

    # Chronological ordering makes IDs independent of source file ordering in
    # downstream exports (the ID itself is content/date-derived).
    merged_records.sort(key=lambda r: (r["review_date"], r["review_id"]))
    return merged_records


def add_relevance_scores(
    df: pd.DataFrame,
    half_life_days: int = RECENCY_HALF_LIFE_DAYS,
    recency_weight: float = RECENCY_WEIGHT,
    helpfulness_weight: float = HELPFULNESS_WEIGHT,
) -> pd.DataFrame:
    """Add lightweight recency/helpfulness ranking columns."""
    if half_life_days <= 0:
        raise ValueError("RECENCY_HALF_LIFE_DAYS must be greater than zero.")
    if recency_weight < 0 or helpfulness_weight < 0:
        raise ValueError("Relevance weights cannot be negative.")
    if not np.isclose(recency_weight + helpfulness_weight, 1.0):
        raise ValueError("RECENCY_WEIGHT and HELPFULNESS_WEIGHT must sum to 1.0.")

    result = df.copy()
    result["review_date"] = pd.to_datetime(result["review_date"], errors="coerce")
    result["developer_reply_date"] = pd.to_datetime(
        result["developer_reply_date"], errors="coerce"
    )

    today = pd.Timestamp.today().normalize()
    result["age_days"] = (today - result["review_date"]).dt.days.clip(lower=0)
    result["recency_weight"] = np.exp(
        -np.log(2) * result["age_days"] / half_life_days
    )

    votes = pd.to_numeric(result["helpful_count"], errors="coerce").clip(lower=0)
    known_votes = votes.notna()
    result["helpfulness_weight"] = np.nan

    if known_votes.any():
        log_votes = np.log1p(votes.loc[known_votes])
        maximum = float(log_votes.max())
        if maximum > 0:
            result.loc[known_votes, "helpfulness_weight"] = log_votes / maximum
        else:
            result.loc[known_votes, "helpfulness_weight"] = 0.0

    # Missing helpful counts remain missing. Those reviews use the recency
    # score alone instead of treating missing votes as zero votes.
    result["priority_score"] = result["recency_weight"]
    result.loc[known_votes, "priority_score"] = (
        recency_weight * result.loc[known_votes, "recency_weight"]
        + helpfulness_weight * result.loc[known_votes, "helpfulness_weight"]
    )

    result["has_developer_reply"] = (
        result["developer_reply_text"].fillna("").astype(str).str.strip().ne("")
    )
    reply_delay = (result["developer_reply_date"] - result["review_date"]).dt.days
    result["reply_delay_days"] = reply_delay.where(reply_delay >= 0)

    return result


def write_summary(
    output_path: Path,
    input_stats: dict[str, dict[str, int]],
    occurrence_count: int,
    unique_count: int,
    issue_count: int,
    df: pd.DataFrame,
) -> None:
    lines = [
        "Google Play review preprocessing summary",
        "=" * 40,
        f"Run date: {pd.Timestamp.today().date().isoformat()}",
        f"Input review occurrences parsed: {occurrence_count}",
        f"Unique reviews after deduplication: {unique_count}",
        f"Duplicates merged: {occurrence_count - unique_count}",
        f"Parsing issues logged: {issue_count}",
        f"Reviews with helpful count: {int(df['helpful_count'].notna().sum())}",
        f"Reviews with developer reply: {int(df['has_developer_reply'].sum())}",
        f"Reviews with inferred rating: {int(df['rating'].notna().sum())}",
        "",
        "Ranking configuration",
        f"- Recency half-life: {RECENCY_HALF_LIFE_DAYS} days",
        f"- Recency weight: {RECENCY_WEIGHT:.2f}",
        f"- Helpfulness weight: {HELPFULNESS_WEIGHT:.2f}",
        "- When helpful votes are missing, priority_score uses recency_weight alone.",
        "- Helpful counts across duplicate snapshots use the maximum observed value;",
        "  all distinct observed counts are retained in helpful_counts_observed.",
        "",
        "Per-file counts",
    ]

    for filename in INPUT_FILES:
        stats = input_stats.get(filename)
        lines.append(f"\n{filename}")
        if not stats:
            lines.append("  Not processed")
            continue
        for key, value in stats.items():
            lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "Important interpretation notes",
        "- priority_score is a heuristic for review prioritization, not a sentiment score.",
        "- No helpful count is represented as missing, not zero.",
        "- Developer replies are kept separate from user review text.",
        "- Ratings are inferred from filenames only; most_relevant.txt alone does not identify rating.",
        "- Rating conflicts across source files are marked in rating_conflict.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / OUTPUT_DIR_NAME

    if not DEVELOPER_ALIASES or not any(alias.strip() for alias in DEVELOPER_ALIASES):
        print("ERROR: Add the app developer's displayed name to DEVELOPER_ALIASES.", file=sys.stderr)
        return 2

    input_paths = [script_dir / filename for filename in INPUT_FILES]
    missing = [path.name for path in input_paths if not path.is_file()]
    if missing:
        print("ERROR: The following input files were not found beside this script:", file=sys.stderr)
        for filename in missing:
            print(f"  - {filename}", file=sys.stderr)
        print(f"\nPut the five .txt files in: {script_dir}", file=sys.stderr)
        return 2

    all_records: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    input_stats: dict[str, dict[str, int]] = {}

    for path in input_paths:
        try:
            records, issues, stats = parse_one_file(path, DEVELOPER_ALIASES)
        except (OSError, ValueError) as exc:
            print(f"ERROR while processing {path.name}: {exc}", file=sys.stderr)
            return 1
        all_records.extend(records)
        all_issues.extend(issues)
        input_stats[path.name] = stats
        print(
            f"Processed {path.name}: {stats['user_reviews_parsed']} reviews, "
            f"{stats['developer_blocks_detected']} developer blocks, "
            f"{len(issues)} parsing issue(s)."
        )

    if not all_records:
        print("ERROR: No user reviews were parsed from the input files.", file=sys.stderr)
        return 1

    occurrence_count = len(all_records)
    merged_records = merge_duplicate_reviews(all_records)
    df = pd.DataFrame(merged_records)

    # Normalize nullable numeric/date columns before exporting.
    df["helpful_count"] = pd.to_numeric(df["helpful_count"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["developer_reply_date"] = pd.to_datetime(df["developer_reply_date"], errors="coerce")
    df = add_relevance_scores(df)

    # Put commonly used columns first; preserve raw text/audit columns at the end.
    preferred_columns = [
        "review_id",
        "review_date",
        "rating",
        "rating_conflict",
        "reviewer_name",
        "review_text_clean",
        "helpful_count",
        "helpful_counts_observed",
        "developer_reply_author",
        "developer_reply_date",
        "developer_reply_text",
        "has_developer_reply",
        "reply_delay_days",
        "age_days",
        "recency_weight",
        "helpfulness_weight",
        "priority_score",
        "source_files",
        "source_count",
        "source_ratings",
        "source_occurrences",
        "review_fingerprint",
        "raw_review_block",
    ]
    column_order = [c for c in preferred_columns if c in df.columns]
    column_order += [c for c in df.columns if c not in column_order]
    df = df[column_order]

    # The preprocessed output is chronological; the ranked copy prioritizes score.
    df_chronological = df.sort_values(
        ["review_date", "review_id"], ascending=[False, True]
    ).reset_index(drop=True)
    df_ranked = df.sort_values(
        ["priority_score", "review_date"], ascending=[False, False]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "reviews_preprocessed.csv"
    ranked_csv = output_dir / "reviews_ranked.csv"
    issues_csv = output_dir / "parsing_issues.csv"
    summary_path = output_dir / "processing_summary.txt"

    df_chronological.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df_ranked.to_csv(ranked_csv, index=False, encoding="utf-8-sig")

    issue_columns = ["source_file", "issue_type", "author", "entry_date", "text"]
    issues_df = pd.DataFrame(all_issues, columns=issue_columns)
    issues_df.to_csv(issues_csv, index=False, encoding="utf-8-sig")

    write_summary(
        summary_path,
        input_stats,
        occurrence_count,
        len(df),
        len(all_issues),
        df,
    )

    print("\nFinished.")
    print(f"Input review occurrences: {occurrence_count}")
    print(f"Unique reviews after deduplication: {len(df)}")
    print(f"Duplicates merged: {occurrence_count - len(df)}")
    print(f"Output folder: {output_dir}")
    print(f"  - {output_csv.name}")
    print(f"  - {ranked_csv.name}")
    print(f"  - {issues_csv.name}")
    print(f"  - {summary_path.name}")
    print("\nCheck parsing_issues.csv and manually verify a sample before analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
