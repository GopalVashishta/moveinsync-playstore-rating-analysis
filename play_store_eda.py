#!/usr/bin/env python3
"""Comprehensive exploratory data analysis for processed Google Play reviews.

Designed for the output schema from preprocess_play_store_reviews.py.
No NLP model, internet access, or seaborn is required.

Default input:
  ./processed_reviews/reviews_preprocessed.csv
Fallback input:
  ./reviews_preprocessed.csv

Outputs a CSV/PNG/TXT EDA bundle under ./play_store_eda_output/.

Important interpretation notes:
- This dataset combines sources sampled differently ("most relevant" and
  rating-specific newest reviews). Combined date trends can reflect collection
  design, not changes in all Play Store reviews.
- The script does not impute unknown ratings or helpful-vote counts.
- Keyword counts and generic-review flags are descriptive heuristics, not topic
  or sentiment labels.
- Monthly outliers describe collected-review volume within a source only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STOPWORDS = set("""
a an the and or but if then else when where while of at by for from in into on onto to up down
with without about against between through during before after above below is am are was were be
been being have has had do does did doing it its this that these those i me my mine we us our ours
you your yours he him his she her hers they them their theirs who whom which what where why how
not no nor so too very can could should would will just than also app application apps use used using
get gets got make makes made one really much many more most some any all only even still already
please hi hello thanks thank thankyou good great nice best worst bad ok okay love hate work works
working worked dont doesn't didnt cant cannot won't wasnt isnt arent im ive you're theyre its you're
""".split())

GENERIC_REVIEW_PATTERNS = [
    r"good", r"very good", r"nice", r"excellent", r"awesome", r"great", r"best",
    r"good app", r"nice app", r"excellent app", r"awesome app", r"very nice",
    r"bad", r"worst", r"poor", r"useless", r"waste of time", r"waste of money",
    r"love it", r"i love this app", r"hate it", r"superb", r"ok", r"okay",
]
GENERIC_RE = re.compile(r"^(?:" + "|".join(GENERIC_REVIEW_PATTERNS) + r")[.!\s🙏👍❤️😍😊😡😂]*$", re.I)
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_sources(value: Any) -> list[str]:
    text = safe_str(value)
    if not text:
        return ["Unknown source"]
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts or ["Unknown source"]


def month_label(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M").astype(str)


def word_tokens(text: Any) -> list[str]:
    return [t.casefold() for t in WORD_RE.findall(safe_str(text))]


def rating_label(value: Any) -> str:
    if pd.isna(value):
        return "Unknown rating"
    try:
        return f"{int(value)}★"
    except (ValueError, TypeError):
        return "Unknown rating"


def fmt_num(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3f}" if abs(value) < 1000 else f"{value:,.1f}"
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_col = "review_text_clean" if "review_text_clean" in out else (
        "review_text" if "review_text" in out else "review"
    )
    if text_col not in out:
        raise ValueError(
            "Could not find review text. Expected one of: review_text_clean, review_text, review."
        )
    out["_analysis_text"] = out[text_col].fillna("").astype(str).str.strip()
    out["_text_chars"] = out["_analysis_text"].str.len()
    out["_tokens"] = out["_analysis_text"].map(word_tokens)
    out["_word_count"] = out["_tokens"].map(len)
    out["_unique_word_count"] = out["_tokens"].map(lambda ts: len(set(ts)))
    out["_text_empty"] = out["_analysis_text"].eq("")
    out["_symbol_only"] = out["_analysis_text"].ne("") & ~out["_analysis_text"].str.contains(
        r"[A-Za-z0-9]", regex=True, na=False
    )
    out["_generic_short_review"] = out["_analysis_text"].str.casefold().str.strip().map(
        lambda s: bool(GENERIC_RE.fullmatch(s)) if s else False
    )
    out["_month"] = month_label(out["review_date"])
    out["_rating_label"] = out["rating"].map(rating_label) if "rating" in out else "Unknown rating"
    out["_source_list"] = out["source_files"].map(parse_sources) if "source_files" in out else [["Unknown source"]] * len(out)
    out["_source_count_calc"] = out["_source_list"].map(len)
    out["_reviewer_key"] = out["reviewer_name"].fillna("Unknown reviewer").astype(str) if "reviewer_name" in out else "Unknown reviewer"
    return out


def robust_outlier_flags(monthly: pd.DataFrame, group_cols: list[str], value_col: str = "review_count") -> pd.DataFrame:
    """Flag high/low collection-count months using median +/- 3*MAD; descriptive only."""
    if monthly.empty:
        return pd.DataFrame(columns=[*group_cols, "month", value_col, "median", "mad", "robust_z", "outlier_flag"])
    rows = []
    for keys, g in monthly.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(g[value_col], errors="coerce")
        med = values.median()
        mad = (values - med).abs().median()
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        for _, row in g.iterrows():
            rz = np.nan
            method = "insufficient_months"
            flag = False
            # Need at least six months before flagging a descriptive count anomaly.
            if len(values) >= 6:
                if pd.notna(mad) and mad > 0:
                    rz = 0.6745 * (row[value_col] - med) / mad
                    flag = abs(rz) >= 3.5
                    method = "median_mad"
                elif pd.notna(iqr) and iqr > 0:
                    flag = (row[value_col] < q1 - 1.5 * iqr) or (row[value_col] > q3 + 1.5 * iqr)
                    method = "iqr_fallback"
                elif pd.notna(med):
                    # Flat/zero-inflated series: flag only deviations above a flat median.
                    flag = row[value_col] > med
                    method = "flat_series_deviation"
            record = dict(zip(group_cols, keys))
            record.update({"month": row["month"], value_col: row[value_col], "median": med,
                           "mad": mad, "robust_z": rz, "outlier_flag": bool(flag),
                           "method": method, "months_in_source_window": len(values)})
            rows.append(record)
    return pd.DataFrame(rows)


def plot_bar(labels, values, title, xlabel, ylabel, path, rotation=25, horizontal=False):
    plt.figure(figsize=(10, 5.8))
    if horizontal:
        order = np.argsort(np.asarray(values))
        labels = np.asarray(labels)[order]
        values = np.asarray(values)[order]
        plt.barh(labels, values)
    else:
        plt.bar(labels, values)
        plt.xticks(rotation=rotation, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to reviews_preprocessed.csv (default searches standard locations).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: ./play_store_eda_output).")
    parser.add_argument("--top-terms", type=int, default=25,
                        help="Number of frequent unigrams and bigrams to export/plot (default: 25).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    input_path = args.input
    if input_path is None:
        candidates = [
            here / "processed_reviews" / "reviews_preprocessed.csv",
            here / "reviews_preprocessed.csv",
            Path.cwd() / "processed_reviews" / "reviews_preprocessed.csv",
            Path.cwd() / "reviews_preprocessed.csv",
        ]
        input_path = next((p for p in candidates if p.is_file()), None)
    if input_path is None or not Path(input_path).is_file():
        print("ERROR: Could not locate reviews_preprocessed.csv. Use --input PATH or place it in", file=sys.stderr)
        print("./processed_reviews/reviews_preprocessed.csv relative to this script.", file=sys.stderr)
        return 2

    output_dir = args.output_dir or (here / "play_store_eda_output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    plots = output_dir / "plots"

    try:
        raw = pd.read_csv(input_path, low_memory=False)
    except Exception as exc:
        print(f"ERROR reading {input_path}: {exc}", file=sys.stderr)
        return 1

    if raw.empty:
        print("ERROR: Input CSV has headers but no rows.", file=sys.stderr)
        return 1

    required_review_text = next((c for c in ["review_text_clean", "review_text", "review"] if c in raw.columns), None)
    if required_review_text is None:
        print("ERROR: Input CSV must contain review_text_clean (or review_text/review).", file=sys.stderr)
        print("Available columns: " + ", ".join(raw.columns), file=sys.stderr)
        return 1

    df = raw.copy()
    for col in ["review_date", "developer_reply_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
        else:
            df[col] = pd.NaT
    for col in ["rating", "helpful_count", "priority_score", "reply_delay_days", "source_count", "source_occurrences"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rating_conflict" not in df:
        df["rating_conflict"] = False
    df["rating_conflict"] = df["rating_conflict"].astype(str).str.casefold().isin(["true", "1", "yes"])
    if "developer_reply_text" not in df:
        df["developer_reply_text"] = np.nan
    df["has_developer_reply"] = df["developer_reply_text"].fillna("").astype(str).str.strip().ne("")
    if "helpful_count" not in df:
        df["helpful_count"] = np.nan
    if "review_id" not in df:
        df["review_id"] = [f"ROW_{i+1}" for i in range(len(df))]
    df = add_feature_columns(df)

    # Basic data-quality and sample-description tables.
    n = len(df)
    date_valid = df["review_date"].notna()
    rating_valid = df["rating"].isin([1, 2, 3, 4, 5])
    helpful_valid = df["helpful_count"].notna()
    overview = [
        ("rows", n, "Input rows, usually deduplicated reviews"),
        ("unique_review_id", df["review_id"].nunique(dropna=True), "Unique values in review_id"),
        ("duplicate_review_id_rows", int(df["review_id"].duplicated(keep=False).sum()), "Rows whose review_id occurs more than once"),
        ("valid_review_dates", int(date_valid.sum()), "Rows with parseable review_date"),
        ("missing_review_dates", int((~date_valid).sum()), "Rows with missing/unparseable review_date"),
        ("min_review_date", df.loc[date_valid, "review_date"].min() if date_valid.any() else pd.NaT, "Earliest review date in the collected sample"),
        ("max_review_date", df.loc[date_valid, "review_date"].max() if date_valid.any() else pd.NaT, "Latest review date in the collected sample"),
        ("valid_1_to_5_star_ratings", int(rating_valid.sum()), "Rows with rating in 1..5; unknowns are not imputed"),
        ("missing_or_out_of_range_ratings", int((~rating_valid).sum()), "Missing or invalid star rating rows"),
        ("rating_conflict_rows", int(df["rating_conflict"].sum()), "Rows flagged for conflicting source ratings"),
        ("reviews_with_helpful_count", int(helpful_valid.sum()), "Rows with observed helpful vote count"),
        ("missing_helpful_count_rows", int((~helpful_valid).sum()), "Missing vote count; not treated as zero"),
        ("reviews_with_developer_reply", int(df["has_developer_reply"].sum()), "Rows with nonempty developer_reply_text"),
        ("reviews_without_developer_reply", int((~df["has_developer_reply"]).sum()), "No captured reply; not proof that company never replied"),
        ("empty_review_text_rows", int(df["_text_empty"].sum()), "Rows with empty cleaned review text"),
        ("symbol_only_text_rows", int(df["_symbol_only"].sum()), "Nonempty text containing no ASCII letters/digits (approximate heuristic)"),
        ("generic_short_text_rows", int(df["_generic_short_review"].sum()), "Exact-match heuristic for generic short comments; do not auto-delete"),
        ("reviews_from_multiple_sources", int((df["_source_count_calc"] > 1).sum()), "Rows whose source_files lists multiple input files"),
        ("max_source_occurrences", pd.to_numeric(df["source_occurrences"], errors="coerce").max() if "source_occurrences" in df else np.nan, "Largest saved source_occurrences value, when available"),
        ("median_review_chars", df["_text_chars"].median(), "Median character count in cleaned review text"),
        ("median_review_words", df["_word_count"].median(), "Median word count in cleaned review text"),
        ("median_helpful_count_among_observed", df.loc[helpful_valid, "helpful_count"].median() if helpful_valid.any() else np.nan, "Median observed helpful votes; not adjusted for review age"),
    ]
    overview_df = pd.DataFrame(overview, columns=["metric", "value", "interpretation"])
    save_csv(overview_df, output_dir / "overview_metrics.csv")

    # Missingness.
    audit_cols = [c for c in ["review_date", "rating", "review_text_clean", "helpful_count", "developer_reply_text", "developer_reply_date", "source_files", "source_count", "source_occurrences", "priority_score"] if c in df.columns]
    missing_df = pd.DataFrame({
        "column": audit_cols,
        "missing_count": [int(df[c].isna().sum()) + (df[c].astype(str).str.strip().eq("").sum() if df[c].dtype == "object" else 0) for c in audit_cols],
    })
    missing_df["missing_pct"] = missing_df["missing_count"] / n * 100
    save_csv(missing_df, output_dir / "missingness.csv")

    # Rating breakdown; do not infer/repair ratings here.
    df["rating_group"] = df["_rating_label"]
    rating_summary = df.groupby("rating_group", dropna=False).agg(
        review_count=("review_id", "size"),
        pct_of_dataset=("review_id", lambda s: len(s) / n * 100),
        median_words=("_word_count", "median"),
        median_chars=("_text_chars", "median"),
        helpful_count_coverage_pct=("helpful_count", lambda s: s.notna().mean() * 100),
        reply_coverage_pct=("has_developer_reply", "mean"),
        rating_conflict_count=("rating_conflict", "sum"),
    ).reset_index()
    save_csv(rating_summary, output_dir / "rating_summary.csv")
    r_order = [f"{i}★" for i in range(1, 6)] + ["Unknown rating"]
    rating_plot = rating_summary.set_index("rating_group").reindex(r_order).dropna(subset=["review_count"])
    plot_bar(rating_plot.index.tolist(), rating_plot["review_count"].tolist(),
             "Review sample size by recorded star rating", "Recorded rating", "Reviews",
             plots / "01_rating_distribution.png", rotation=0)

    # Source summary. Each review is counted once per source membership, so totals across source rows can exceed N.
    src = df[["review_id", "_source_list", "rating_group", "review_date", "helpful_count", "has_developer_reply"]].explode("_source_list").rename(columns={"_source_list": "source_file"})
    source_summary = src.groupby("source_file", dropna=False).agg(
        unique_reviews=("review_id", "nunique"),
        median_review_date=("review_date", "median"),
        earliest_review_date=("review_date", "min"),
        latest_review_date=("review_date", "max"),
        rating_unknown_pct=("rating_group", lambda s: s.eq("Unknown rating").mean() * 100),
        helpful_count_coverage_pct=("helpful_count", lambda s: s.notna().mean() * 100),
        reply_coverage_pct=("has_developer_reply", "mean"),
    ).reset_index()
    source_summary["unique_reviews"] = source_summary["unique_reviews"].astype(int)
    save_csv(source_summary, output_dir / "source_summary.csv")
    source_summary_sorted = source_summary.sort_values("unique_reviews")
    plot_bar(source_summary_sorted["source_file"].tolist(), source_summary_sorted["unique_reviews"].tolist(),
             "Unique reviews represented by source file", "Source file", "Reviews (unique within source)",
             plots / "02_source_volume.png", horizontal=True)

    # Source overlap.
    overlap = df["_source_count_calc"].value_counts().sort_index().rename_axis("number_of_sources").reset_index(name="review_count")
    overlap["pct_of_unique_reviews"] = overlap["review_count"] / n * 100
    save_csv(overlap, output_dir / "source_overlap.csv")
    plot_bar(overlap["number_of_sources"].astype(str).tolist(), overlap["review_count"].tolist(),
             "Review overlap across source files", "Number of source files containing review", "Unique reviews",
             plots / "03_source_overlap.png", rotation=0)

    # Temporal EDA: fill missing calendar months with zero within the observed date range.
    # Source-level lines are essential because this is a mixture of differently selected samples.
    dated = df.loc[date_valid].copy()
    monthly = pd.DataFrame(columns=["month", "review_count"])
    monthly_rating = pd.DataFrame(columns=["month", "rating_group", "review_count", "pct_of_month"])
    monthly_source = pd.DataFrame(columns=["month", "source_file", "review_count"])
    if not dated.empty:
        dated["month"] = dated["review_date"].dt.to_period("M").astype(str)
        month_range = pd.period_range(
            dated["review_date"].min().to_period("M"),
            dated["review_date"].max().to_period("M"), freq="M"
        ).astype(str).tolist()

        monthly_observed = dated.groupby("month").agg(review_count=("review_id", "nunique"))
        monthly = monthly_observed.reindex(month_range, fill_value=0).rename_axis("month").reset_index()
        save_csv(monthly, output_dir / "monthly_review_counts_overall.csv")
        plt.figure(figsize=(12, 5.8))
        plt.plot(monthly["month"], monthly["review_count"], marker="o")
        plt.title("Collected review dates by month — all sources combined")
        plt.xlabel("Review month")
        plt.ylabel("Unique reviews")
        plt.xticks(rotation=45, ha="right")
        save_fig(plots / "04_monthly_counts_overall.png")

        rating_order_present = [x for x in [f"{i}★" for i in range(1, 6)] + ["Unknown rating"] if x in dated["_rating_label"].unique()]
        mr_obs = dated.groupby(["month", "_rating_label"]).agg(review_count=("review_id", "nunique"))
        full_index = pd.MultiIndex.from_product([month_range, rating_order_present], names=["month", "_rating_label"])
        mr = mr_obs.reindex(full_index, fill_value=0).reset_index().rename(columns={"_rating_label": "rating_group"})
        totals = mr.groupby("month")["review_count"].transform("sum")
        mr["pct_of_month"] = np.where(totals > 0, mr["review_count"] / totals * 100, np.nan)
        monthly_rating = mr
        save_csv(monthly_rating, output_dir / "monthly_counts_by_rating.csv")
        plt.figure(figsize=(12, 6.2))
        for label, g in monthly_rating.groupby("rating_group"):
            g = g.sort_values("month")
            plt.plot(g["month"], g["review_count"], marker="o", label=label)
        plt.title("Collected review dates by month and recorded rating")
        plt.xlabel("Review month")
        plt.ylabel("Unique reviews")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Rating")
        save_fig(plots / "05_monthly_counts_by_rating.png")

        plt.figure(figsize=(12, 6.2))
        for label, g in monthly_rating.groupby("rating_group"):
            g = g.sort_values("month")
            plt.plot(g["month"], g["pct_of_month"], marker="o", label=label)
        plt.title("Monthly composition by recorded rating (percent of collected reviews that month)")
        plt.xlabel("Review month")
        plt.ylabel("Percent of monthly sample")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Rating")
        save_fig(plots / "06_monthly_rating_mix_pct.png")

        src_dated = dated[["review_id", "review_date", "month", "rating_group", "_source_list"]].explode("_source_list").rename(columns={"_source_list": "source_file"})
        # Give each source its own observed review-date window. Do not pad a source
        # with zeros from months before its earliest or after its latest collected review.
        monthly_source_parts = []
        for source_name, sg in src_dated.groupby("source_file", dropna=False):
            source_min = sg["review_date"].min().to_period("M")
            source_max = sg["review_date"].max().to_period("M")
            source_months = pd.period_range(source_min, source_max, freq="M").astype(str).tolist()
            source_obs = sg.groupby("month").agg(review_count=("review_id", "nunique"))
            source_monthly = source_obs.reindex(source_months, fill_value=0).rename_axis("month").reset_index()
            source_monthly["source_file"] = source_name
            monthly_source_parts.append(source_monthly[["month", "source_file", "review_count"]])
        monthly_source = pd.concat(monthly_source_parts, ignore_index=True) if monthly_source_parts else pd.DataFrame(columns=["month", "source_file", "review_count"])
        monthly_source = monthly_source.sort_values(["source_file", "month"])
        save_csv(monthly_source, output_dir / "monthly_counts_by_source.csv")
        plt.figure(figsize=(12, 6.2))
        for label, g in monthly_source.groupby("source_file"):
            g = g.sort_values("month")
            plt.plot(g["month"], g["review_count"], marker="o", label=label)
        plt.title("Collected review dates by month and source file")
        plt.xlabel("Review month")
        plt.ylabel("Reviews represented in source (overlap allowed)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Source", fontsize=8)
        save_fig(plots / "07_monthly_counts_by_source.png")

        # Monthly response coverage (descriptive; not a measure of support quality).
        dated_replies = dated.copy()
        dated_replies["month"] = dated_replies["review_date"].dt.to_period("M").astype(str)
        monthly_reply = dated_replies.groupby("month").agg(
            review_count=("review_id", "size"),
            replied_count=("has_developer_reply", "sum"),
            reply_coverage_pct=("has_developer_reply", lambda x: x.mean() * 100),
        ).reindex(month_range)
        monthly_reply.index.name = "month"
        monthly_reply = monthly_reply.reset_index()
        save_csv(monthly_reply, output_dir / "monthly_developer_reply_coverage.csv")
        plt.figure(figsize=(12, 5.5))
        plt.plot(monthly_reply["month"], monthly_reply["reply_coverage_pct"], marker="o")
        plt.title("Captured developer-reply coverage by review month")
        plt.xlabel("Review month")
        plt.ylabel("Reviews with captured reply (%)")
        plt.xticks(rotation=45, ha="right")
        save_fig(plots / "08_monthly_reply_coverage.png")

        # Anomalies flag collected volume changes only within each source; zero months are retained.
        outlier_source = monthly_source.rename(columns={"source_file": "source"})
        outliers = robust_outlier_flags(outlier_source, ["source"])
        save_csv(outliers, output_dir / "monthly_collection_volume_outliers_by_source.csv")
    else:
        save_csv(monthly, output_dir / "monthly_review_counts_overall.csv")
        save_csv(monthly_rating, output_dir / "monthly_counts_by_rating.csv")
        save_csv(monthly_source, output_dir / "monthly_counts_by_source.csv")
        save_csv(pd.DataFrame(), output_dir / "monthly_collection_volume_outliers_by_source.csv")
        save_csv(pd.DataFrame(), output_dir / "monthly_developer_reply_coverage.csv")

    # Rating/source composition table.
    source_rating = src.groupby(["source_file", "rating_group"]).agg(review_count=("review_id", "nunique")).reset_index()
    save_csv(source_rating, output_dir / "source_rating_composition.csv")
    pivot = source_rating.pivot(index="source_file", columns="rating_group", values="review_count").fillna(0)
    if not pivot.empty:
        ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
        ax.set_title("Recorded rating mix within each source file")
        ax.set_xlabel("Source file")
        ax.set_ylabel("Unique reviews (overlap across sources possible)")
        plt.xticks(rotation=25, ha="right")
        plt.legend(title="Rating", fontsize=8)
        save_fig(plots / "09_source_rating_composition.png")

    # Helpful-vote distributions and relation to rating/date/text size.
    observed_votes = df.loc[df["helpful_count"].notna() & (df["helpful_count"] >= 0)].copy()
    helpful_stats = []
    if not observed_votes.empty:
        vv = observed_votes["helpful_count"]
        for metric, value, note in [
            ("observed_vote_rows", len(vv), "Only reviews with a captured vote count"),
            ("mean_helpful_count", vv.mean(), "Right-skew can make mean exceed typical values"),
            ("median_helpful_count", vv.median(), "Median of observed helpful counts"),
            ("p75_helpful_count", vv.quantile(.75), "75th percentile among observed counts"),
            ("p90_helpful_count", vv.quantile(.90), "90th percentile among observed counts"),
            ("p95_helpful_count", vv.quantile(.95), "95th percentile among observed counts"),
            ("max_helpful_count", vv.max(), "Maximum captured vote count"),
            ("zero_vote_pct_among_observed", (vv == 0).mean() * 100, "Only meaningful if zeros were explicitly observed, not missing"),
        ]:
            helpful_stats.append({"metric": metric, "value": value, "interpretation": note})
        plt.figure(figsize=(9, 5.5))
        plt.hist(np.log1p(vv), bins=35)
        plt.title("Distribution of observed helpful votes (log1p scale)")
        plt.xlabel("log(1 + helpful_count)")
        plt.ylabel("Reviews")
        save_fig(plots / "10_helpful_votes_log_histogram.png")

        by_rating_votes = observed_votes.copy()
        by_rating_votes["rating_group"] = by_rating_votes["_rating_label"]
        grouped = by_rating_votes.groupby("rating_group")["helpful_count"]
        ordered_labels = [x for x in [f"{i}★" for i in range(1, 6)] + ["Unknown rating"] if x in grouped.groups]
        data = [np.log1p(grouped.get_group(x).clip(lower=0).values) for x in ordered_labels if len(grouped.get_group(x))]
        labels = [x for x in ordered_labels if len(grouped.get_group(x))]
        if data:
            plt.figure(figsize=(10, 5.5))
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.title("Observed helpful votes by rating (log1p; outliers hidden for readability)")
            plt.xlabel("Recorded rating")
            plt.ylabel("log(1 + helpful_count)")
            save_fig(plots / "11_helpful_votes_by_rating.png")

        # Vote count vs review age: descriptive scatter, sample source and exposure effects still apply.
        scatter = observed_votes.loc[observed_votes["review_date"].notna()].copy()
        if not scatter.empty:
            scatter["age_days_at_analysis"] = (pd.Timestamp.today().normalize() - scatter["review_date"]).dt.days.clip(lower=0)
            plt.figure(figsize=(9, 5.5))
            plt.scatter(scatter["age_days_at_analysis"], np.log1p(scatter["helpful_count"]), alpha=.45, s=16)
            plt.title("Helpful votes vs review age — exposure confounding expected")
            plt.xlabel("Days since review date (at analysis run)")
            plt.ylabel("log(1 + helpful_count)")
            save_fig(plots / "12_helpful_votes_vs_review_age.png")
    else:
        helpful_stats.append({"metric": "observed_vote_rows", "value": 0, "interpretation": "No observed helpful counts; no vote distribution plotted"})
    save_csv(pd.DataFrame(helpful_stats), output_dir / "helpful_vote_summary.csv")

    help_by_rating = df.groupby("rating_group").agg(
        review_count=("review_id", "size"),
        vote_count_observed=("helpful_count", "count"),
        vote_coverage_pct=("helpful_count", lambda s: s.notna().mean() * 100),
        median_votes_observed=("helpful_count", "median"),
        mean_votes_observed=("helpful_count", "mean"),
    ).reset_index()
    save_csv(help_by_rating, output_dir / "helpful_vote_coverage_by_rating.csv")

    # Developer reply coverage and delay.
    reply_by_rating = df.groupby("rating_group").agg(
        review_count=("review_id", "size"),
        replied_reviews=("has_developer_reply", "sum"),
        reply_coverage_pct=("has_developer_reply", lambda s: s.mean() * 100),
    ).reset_index()
    save_csv(reply_by_rating, output_dir / "developer_reply_coverage_by_rating.csv")
    plot_bar(reply_by_rating["rating_group"].tolist(), reply_by_rating["reply_coverage_pct"].tolist(),
             "Captured developer-reply coverage by rating", "Recorded rating", "Reviews with reply (%)",
             plots / "13_reply_coverage_by_rating.png", rotation=0)

    reply_dated = df.loc[df["has_developer_reply"] & df["review_date"].notna() & df["developer_reply_date"].notna()].copy()
    if not reply_dated.empty:
        reply_dated["reply_delay_recomputed_days"] = (reply_dated["developer_reply_date"] - reply_dated["review_date"]).dt.total_seconds() / 86400
        delays = reply_dated["reply_delay_recomputed_days"]
        delay_summary = pd.DataFrame([{
            "replies_with_both_dates": len(reply_dated),
            "negative_delay_count": int((delays < 0).sum()),
            "same_day_reply_count": int((delays == 0).sum()),
            "median_delay_days_nonnegative": delays[delays >= 0].median(),
            "p75_delay_days_nonnegative": delays[delays >= 0].quantile(.75),
            "p90_delay_days_nonnegative": delays[delays >= 0].quantile(.90),
            "max_delay_days_nonnegative": delays[delays >= 0].max(),
        }])
        save_csv(delay_summary, output_dir / "reply_delay_summary.csv")
        nonnegative = delays[delays >= 0]
        if not nonnegative.empty:
            plt.figure(figsize=(9, 5.5))
            plt.hist(nonnegative, bins=min(35, max(5, int(np.sqrt(len(nonnegative))))))
            plt.title("Developer reply delay (days; nonnegative delays only)")
            plt.xlabel("Days from review to captured developer reply")
            plt.ylabel("Reviews")
            save_fig(plots / "14_reply_delay_histogram.png")
    else:
        save_csv(pd.DataFrame([{"replies_with_both_dates": 0, "note": "No reply records with both dates available"}]), output_dir / "reply_delay_summary.csv")

    # Text size and lexical descriptives.
    text_summary = df.groupby("rating_group").agg(
        review_count=("review_id", "size"),
        mean_chars=("_text_chars", "mean"),
        median_chars=("_text_chars", "median"),
        p90_chars=("_text_chars", lambda s: s.quantile(.90)),
        mean_words=("_word_count", "mean"),
        median_words=("_word_count", "median"),
        p90_words=("_word_count", lambda s: s.quantile(.90)),
        generic_short_count=("_generic_short_review", "sum"),
        empty_count=("_text_empty", "sum"),
    ).reset_index()
    save_csv(text_summary, output_dir / "review_text_length_by_rating.csv")

    plt.figure(figsize=(9, 5.5))
    plt.hist(df["_word_count"], bins=40)
    plt.title("Review length distribution")
    plt.xlabel("Word count per review")
    plt.ylabel("Reviews")
    save_fig(plots / "15_review_word_count_distribution.png")

    order = [x for x in [f"{i}★" for i in range(1, 6)] + ["Unknown rating"] if x in df["rating_group"].unique()]
    boxdata = [df.loc[df["rating_group"] == r, "_word_count"].values for r in order]
    if boxdata:
        plt.figure(figsize=(10, 5.5))
        plt.boxplot(boxdata, tick_labels=order, showfliers=False)
        plt.title("Review word count by recorded rating (outliers hidden for readability)")
        plt.xlabel("Recorded rating")
        plt.ylabel("Words per review")
        save_fig(plots / "16_review_length_by_rating.png")

    # Common unigrams / bigrams excluding simple stopwords. Descriptive only, not semantic themes.
    unigram_counts: dict[str, int] = {}
    bigram_counts: dict[str, int] = {}
    for tokens in df["_tokens"]:
        filtered = [t for t in tokens if t not in STOPWORDS and not t.isnumeric() and len(t) > 1]
        for token in filtered:
            unigram_counts[token] = unigram_counts.get(token, 0) + 1
        for a, b in zip(filtered, filtered[1:]):
            pair = f"{a} {b}"
            bigram_counts[pair] = bigram_counts.get(pair, 0) + 1
    top_n = max(5, args.top_terms)
    unigrams_df = pd.DataFrame(sorted(unigram_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n], columns=["term", "review_occurrences"])
    bigrams_df = pd.DataFrame(sorted(bigram_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n], columns=["bigram", "review_occurrences"])
    save_csv(unigrams_df, output_dir / "common_unigrams.csv")
    save_csv(bigrams_df, output_dir / "common_bigrams.csv")
    if not unigrams_df.empty:
        plot_bar(unigrams_df["term"].tolist()[::-1], unigrams_df["review_occurrences"].tolist()[::-1],
                 f"Top {min(top_n, len(unigrams_df))} unigrams (simple stopword filter)", "Token", "Occurrences", plots / "17_common_unigrams.png", horizontal=True)
    if not bigrams_df.empty:
        plot_bar(bigrams_df["bigram"].tolist()[::-1], bigrams_df["review_occurrences"].tolist()[::-1],
                 f"Top {min(top_n, len(bigrams_df))} bigrams (simple stopword filter)", "Bigram", "Occurrences", plots / "18_common_bigrams.png", horizontal=True)

    # Optional priority score distribution is included as metadata, not treated as sentiment or severity.
    if "priority_score" in df.columns and df["priority_score"].notna().any():
        plt.figure(figsize=(9, 5.5))
        plt.hist(df["priority_score"].dropna(), bins=35)
        plt.title("Preprocessing priority_score distribution (heuristic ranking only)")
        plt.xlabel("priority_score")
        plt.ylabel("Reviews")
        save_fig(plots / "19_priority_score_distribution.png")

    # Relationships: rank correlations are descriptive, not causal or significance tests.
    relation_frame = df.copy()
    relation_frame["_rating_numeric"] = pd.to_numeric(relation_frame.get("rating"), errors="coerce")
    relation_frame["_helpful_numeric"] = pd.to_numeric(relation_frame["helpful_count"], errors="coerce")
    relation_frame["_word_count_numeric"] = pd.to_numeric(relation_frame["_word_count"], errors="coerce")
    if relation_frame["review_date"].notna().any():
        relation_frame["_age_days"] = (pd.Timestamp.today().normalize() - relation_frame["review_date"]).dt.days.clip(lower=0)
    else:
        relation_frame["_age_days"] = np.nan

    pairings = [
        ("recorded_rating", "_rating_numeric", "review_word_count", "_word_count_numeric", "Association between recorded star rating and review length"),
        ("recorded_rating", "_rating_numeric", "helpful_count", "_helpful_numeric", "Association between recorded star rating and observed helpful votes"),
        ("review_age_days", "_age_days", "helpful_count", "_helpful_numeric", "Older reviews have had more time to accumulate helpful votes; exposure confounding expected"),
        ("review_word_count", "_word_count_numeric", "helpful_count", "_helpful_numeric", "Association between review length and observed helpful votes"),
    ]
    relationship_rows = []
    for x_name, x_col, y_name, y_col, interpretation in pairings:
        pair = relation_frame[[x_col, y_col]].dropna()
        if len(pair) >= 3 and pair[x_col].nunique() > 1 and pair[y_col].nunique() > 1:
            rho = pair[x_col].rank(method="average").corr(pair[y_col].rank(method="average"))
        else:
            rho = np.nan
        relationship_rows.append({
            "variable_x": x_name,
            "variable_y": y_name,
            "n_complete_pairs": len(pair),
            "spearman_rho": rho,
            "interpretation_note": interpretation,
        })
    relationship_df = pd.DataFrame(relationship_rows)
    save_csv(relationship_df, output_dir / "relationship_summary.csv")

    # Monthly composition denominator is important: raw rating counts fluctuate with collection volume.
    # A volume outlier is only a point worth checking, never proof of a product event.

    # Enriched row-level export is useful for later sampling / cluster inspection.
    export_cols = [c for c in raw.columns]
    for c in ["_text_chars", "_word_count", "_unique_word_count", "_text_empty", "_symbol_only", "_generic_short_review", "_month", "_rating_label", "_source_count_calc"]:
        export_cols.append(c)
    save_csv(df[export_cols], output_dir / "review_level_eda_features.csv")

    # Compact narrative report containing dataset-grounded descriptions only (no causal inference).
    def metric_value(name: str) -> Any:
        row = overview_df.loc[overview_df["metric"] == name, "value"]
        return row.iloc[0] if len(row) else np.nan
    rating_counts = df["_rating_label"].value_counts(dropna=False).to_dict()
    source_counts = source_summary.sort_values("unique_reviews", ascending=False).head(5)
    top_source_line = "; ".join(f"{r.source_file}: {int(r.unique_reviews)}" for r in source_counts.itertuples())
    month_line = "No parseable review dates."
    if not monthly.empty:
        peak = monthly.loc[monthly["review_count"].idxmax()]
        trough = monthly.loc[monthly["review_count"].idxmin()]
        month_line = f"Highest collected-review month: {peak['month']} ({int(peak['review_count'])}); lowest: {trough['month']} ({int(trough['review_count'])})."
    report = [
        "PLAY STORE REVIEW EDA — DATA PROFILE AND DESCRIPTIVE FINDINGS",
        "=" * 66,
        f"Input: {Path(input_path).resolve()}",
        f"EDA run date: {pd.Timestamp.today().date().isoformat()}",
        "",
        "1) Dataset profile",
        f"- Rows: {n:,}; unique review_id values: {int(df['review_id'].nunique(dropna=True)):,}.",
        f"- Parsed review-date range: {metric_value('min_review_date')} to {metric_value('max_review_date')}.",
        f"- Missing/invalid dates: {int(metric_value('missing_review_dates')):,}.",
        f"- Missing/invalid ratings: {int(metric_value('missing_or_out_of_range_ratings')):,}; rating-conflict rows: {int(metric_value('rating_conflict_rows')):,}.",
        f"- Missing helpful counts: {int(metric_value('missing_helpful_count_rows')):,}; these are not treated as zero votes.",
        f"- Reviews with captured developer replies: {int(metric_value('reviews_with_developer_reply')):,} ({float(metric_value('reviews_with_developer_reply'))/n*100:.1f}% of rows).",
        f"- Median review length: {fmt_num(metric_value('median_review_words'))} words / {fmt_num(metric_value('median_review_chars'))} characters.",
        f"- Multiple-source reviews: {int(metric_value('reviews_from_multiple_sources')):,}.",
        "",
        "2) Rating mix",
        "- Recorded rating counts: " + "; ".join(f"{k}: {int(v):,}" for k, v in sorted(rating_counts.items())),
        "- Read rating charts alongside the unknown/conflicting-rating counts; this EDA does not repair ratings.",
        "",
        "3) Source composition",
        f"- Largest source counts (reviews may occur in more than one source): {top_source_line or 'None'}.",
        "- Totals across source files are not expected to sum to the number of unique reviews because duplicate reviews retain multi-source provenance.",
        "",
        "4) Temporal description",
        f"- {month_line}",
        "- IMPORTANT: overall monthly counts combine a 'most relevant' sample with rating-specific newest samples. Peaks and dips can reflect sampling/collection composition; they are not automatically changes in issue prevalence or app quality.",
        "- Use the source-specific monthly chart before interpreting a time-series pattern. Monthly counts are review dates in the collected sample, not a population estimate of all app reviews.",
        "",
        "5) Helpful votes and company responses",
        "- Helpful-count statistics include only observed values; older reviews have had more time to accrue votes, and source visibility may differ.",
        "- Reply coverage means a developer reply was captured in this dataset. Absence is not proof that no reply exists elsewhere.",
        "- Reply-delay metrics are descriptive and require both dates. Negative delays are flagged, not silently repaired.",
        "",
        "6) Text descriptions",
        f"- Generic-short-comment matches from a small exact-match list: {int(df['_generic_short_review'].sum()):,}. This is a review-for-inspection flag, not automatic deletion.",
        "- Frequent words/bigrams are lexical summaries. They can be distorted by phrasing, negation, spelling variants and words that are not issue categories.",
        "- This EDA does not infer sentiment, diagnose root causes, or identify semantic topic clusters.",
        "",
        "7) Files generated",
        "- overview_metrics.csv, missingness.csv, rating_summary.csv, source_summary.csv, source_overlap.csv",
        "- monthly_review_counts_overall.csv, monthly_counts_by_rating.csv (includes pct_of_month), monthly_counts_by_source.csv, monthly_collection_volume_outliers_by_source.csv, monthly_developer_reply_coverage.csv",
        "- source_rating_composition.csv, helpful_vote_summary.csv, helpful_vote_coverage_by_rating.csv",
        "- developer_reply_coverage_by_rating.csv, reply_delay_summary.csv, review_text_length_by_rating.csv",
        "- common_unigrams.csv, common_bigrams.csv, relationship_summary.csv, review_level_eda_features.csv, plots/*.png",
        "",
        "Interpret findings as descriptions of the collected sample. Check source-specific plots, month-by-month sample sizes, missingness, and rating provenance before treating any apparent relationship as a product-level signal.",
    ]
    (output_dir / "eda_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"EDA complete. Input rows: {n:,}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Report: {(output_dir / 'eda_report.txt').resolve()}")
    print(f"Plots: {plots.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
