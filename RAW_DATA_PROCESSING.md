# Google Play Store Reviews: Data Preprocessing and Relevance Ranking

## 1. Overview

This project preprocesses raw, text-based Google Play Store reviews from five input files, converts them into a structured dataset, removes interface noise, extracts review metadata, associates company responses with reviews, deduplicates overlapping records, and calculates a priority score based on review recency and helpfulness.

The script processes the following input files:

- `most_relevant.txt`
- `newest_1start.txt`
- `newest_2star_apr_12oct.txt`
- `newest_3star_apr_oct.txt`
- `newest_4star_jun_oct.txt`

The objective is to make the raw data suitable for downstream analysis, such as sentiment analysis, complaint classification, topic modeling, feature requests, bug detection, and analysis of company support responses.

The script uses Python, Pandas, and NumPy. It operates locally without requiring API calls or internet access.

## 2. Processing Pipeline

The script follows these steps:

1. Load the five raw text files.
2. Detect review and response blocks using author names and dates.
3. Separate user reviews from developer responses.
4. Remove Google Play interface noise.
5. Extract helpful-vote counts and review dates.
6. Infer star ratings from filenames when possible.
7. Deduplicate overlapping reviews across input files.
8. Combine metadata from duplicate occurrences.
9. Calculate recency, helpfulness, and overall priority scores.
10. Export structured, ranked, and diagnostic output files.

---

## 3. Step-by-Step Data Preprocessing

### Step 1: Load the raw text files

The script reads the five files using UTF-8 encoding, with a fallback to Windows-1252 encoding if necessary.

Each file contains multiple text blocks. A typical block has the following structure:

```text
A Google user
August 26, 2019
The app is difficult to use...
17 people found this review helpful
Did you find this helpful?
MoveInSync: Enabling Hybrid Workplaces
August 26, 2019
We are sorry to hear about your experience...
```

The raw data is not initially treated as a CSV or a table. Instead, the script reconstructs individual entries by identifying author and date lines.

### Step 2: Identify review and response blocks

The script searches for supported date formats, including dates such as:

- `August 26, 2019`
- `Aug 26, 2019`
- `2019-08-26`
- `26 August 2019`

A date is recognized as the beginning of an entry when a non-empty author line appears immediately before it, allowing intervening blank lines.

The text after the date belongs to that entry and ends immediately before the next detected author-and-date pair.

This method separates concatenated entries without relying on a fixed review length.

**Important assumption:** each entry must contain a recognizable author line followed by a supported date line.

### Step 3: Separate user reviews from company responses

The script uses a configurable list of developer-name aliases to distinguish company responses from user reviews.

For example:

```python
DEVELOPER_ALIASES = [
    "MoveInSync",
]
```

If an entry's author matches a configured alias, it is treated as a developer response rather than a user review.

The response is associated with the immediately preceding user review in the same input file, provided that review has not already received a response. Responses that cannot be associated safely are recorded in `parsing_issues.csv`.

Developer responses are stored separately from user review text so that the company's statements do not contaminate sentiment or topic analysis of customer reviews.

### Step 4: Remove Google Play interface noise

The script removes interface text that does not provide meaningful information about the user's experience.

Examples include:

```text
17 people found this review helpful
Did you find this helpful?
Did u find this useful?
```

The helpful-vote count is extracted before the corresponding line is removed.

For example:

**Raw block**

```text
The app crashes whenever I try to book a ride.
17 people found this review helpful
Did you find this helpful?
```

**Cleaned review text**

```text
The app crashes whenever I try to book a ride.
```

**Extracted metadata**

```text
helpful_count = 17
```

This prevents interface prompts and vote counts from being interpreted as part of the customer's actual feedback.

The script collapses repeated whitespace but preserves the original wording, punctuation, and capitalization of the cleaned review text.

The uncleaned entry body is retained in `raw_review_block` for auditing and troubleshooting.

### Step 5: Extract and preserve metadata

Each review is converted into a structured record containing fields such as:

| Field | Description |
|---|---|
| `review_id` | Identifier derived from the review's fingerprint |
| `reviewer_name` | Displayed reviewer name |
| `review_date` | Date on which the review was published |
| `review_text_clean` | Review text with interface noise removed |
| `helpful_count` | Extracted number of helpful votes |
| `developer_reply_author` | Displayed company-response author |
| `developer_reply_date` | Date of the company response |
| `developer_reply_text` | Company's response text |
| `has_developer_reply` | Whether a response is present |
| `raw_review_block` | Original entry body before interface-noise removal |

Missing information is handled explicitly.

- If a helpful-vote count is not present, `helpful_count` remains missing rather than being set to zero.
- If a company response is absent, the response fields remain empty.
- If a response is present, it is retained independently of the review text.

This distinction matters because an unavailable vote count is not evidence that a review received zero votes.

### Step 6: Infer star ratings from filenames

The script attempts to identify the rating associated with a source file using its filename.

For example:

| Filename | Inferred rating |
|---|---:|
| `newest_1start.txt` | 1 star |
| `newest_2star_apr_12oct.txt` | 2 stars |
| `newest_3star_apr_oct.txt` | 3 stars |
| `newest_4star_jun_oct.txt` | 4 stars |
| `most_relevant.txt` | Unknown |

The rating is inferred from the filename, not from the wording or sentiment of the review.

Therefore, rating values should be considered source-level metadata. The script cannot independently verify that every review was placed in the correct source file.

If the same review occurs in sources with different inferred ratings, the script marks `rating_conflict = True` and leaves the consolidated `rating` missing. The source ratings are preserved for investigation.

---

## 4. Deduplication Across Multiple Files

The five input files may contain overlapping reviews. For example, a review may occur in both the most-relevant export and a star-specific export.

Without deduplication, such reviews could appear multiple times in the final analysis and distort review counts, topic frequencies, and other statistics.

### Step 7: Generate a review fingerprint

The script normalizes three fields for matching:

1. Reviewer name
2. Review date
3. Cleaned review text

Normalization includes Unicode normalization, whitespace collapsing, and case-insensitive comparison.

The normalized fields are combined into a single string:

\[
K_i = N_i \; || \; D_i \; || \; T_i
\]

where:

- \(N_i\) is the normalized reviewer name.
- \(D_i\) is the review date in `YYYY-MM-DD` format.
- \(T_i\) is the normalized review text.
- \(||\) denotes concatenation with a separator.

A SHA-256 hash is then calculated, and its first 16 hexadecimal characters are used as the fingerprint:

\[
F_i = \operatorname{SHA256}(K_i)[0:16]
\]

Reviews with identical normalized reviewer names, dates, and text produce the same fingerprint and are grouped together.

**Deduplication limitation:** this is exact normalized-text matching, not semantic matching. Reviews with different wording, punctuation, or dates may remain separate even if they describe the same issue.

### Step 8: Consolidate duplicate records

When duplicates are identified, the script merges their metadata rather than retaining separate review rows.

The following rules are applied:

- **Source files:** combine the unique filenames in which the review appears.
- **Source count:** record the number of distinct input files containing the review.
- **Source occurrences:** count the total number of parsed appearances before deduplication.
- **Helpful votes:** use the maximum observed helpful count across duplicate occurrences.
- **Observed vote counts:** preserve the distinct recorded vote counts in `helpful_counts_observed`.
- **Ratings:** retain the inferred rating if there is only one distinct rating; otherwise, flag the conflict.
- **Developer response:** if multiple responses are available, retain the response with the latest response date.

For example, if the same review appears in two files with helpful counts of 12 and 18, the consolidated record uses 18 for ranking and preserves both values in `helpful_counts_observed`.

The maximum vote count is used as a practical way to avoid counting separate snapshots as independent reviews. It is not necessarily the vote count at any particular date because the counts are cumulative snapshots.

---

## 5. Review Relevance Ranking

After cleaning and deduplication, the script calculates a priority score for each review.

The score combines two signals:

1. **Recency:** newer reviews receive a higher time weight.
2. **Helpfulness:** reviews that other users marked as helpful receive a higher helpfulness weight.

The resulting score is a prioritization heuristic. It is not a sentiment score, a measure of complaint severity, or a prediction of whether the company has fixed an issue.

### Step 9: Calculate review age

The age of each review is calculated using the execution date and its review date.

\[
A_i = \max(0,\ D_{\text{today}} - D_i)
\]

where:

- \(A_i\) is the age of review \(i\), in days.
- \(D_{\text{today}}\) is the current date when the script runs.
- \(D_i\) is the review publication date.

Negative ages are clipped to zero.

Because the calculation uses the run date, recency scores can change when the script is executed again on a later date.

### Step 10: Calculate the recency weight

The script uses exponential decay with a configurable half-life of 730 days.

\[
R_i = e^{-\ln(2)\frac{A_i}{H}}
\]

Equivalently:

\[
\boxed{R_i = 2^{-\frac{A_i}{H}}}
\]

where:

- \(R_i\) is the recency weight.
- \(A_i\) is the review age in days.
- \(H = 730\) days is the configured half-life.

A half-life of 730 days means that a review's recency weight falls to half its original value after 730 days.

Examples:

| Review age | Recency weight |
|---|---:|
| 0 days | 1.000 |
| 365 days | 0.707 |
| 730 days | 0.500 |
| 1,460 days | 0.250 |

These values illustrate the exponential-decay formula; actual weights depend on the execution date and each review's publication date.

The formula prevents older reviews from automatically receiving the same time-based priority as recent reviews while still retaining historical feedback.

### Step 11: Calculate the helpfulness weight

Helpful-vote counts can vary widely. For example, a review with 500 votes should not necessarily receive 50 times the weight of a review with 10 votes.

To reduce the influence of extreme counts, the script applies a natural-log transformation.

First, it calculates:

\[
L_i = \ln(1 + V_i)
\]

where \(V_i\) is the observed helpful-vote count.

Adding 1 allows reviews with zero helpful votes to be represented without taking the logarithm of zero.

The transformed counts are then normalized against the maximum transformed count among reviews with known helpful counts:

\[
\boxed{
U_i =
\frac{\ln(1 + V_i)}
{\max_{j \in S}\ln(1 + V_j)}
}
\]

where:

- \(U_i\) is the normalized helpfulness weight.
- \(V_i\) is the helpful-vote count for review \(i\).
- \(S\) is the set of reviews whose helpful counts are available.
- \(j\) indexes reviews in that set.

When at least one available vote count is positive, the normalized weight ranges from 0 to 1.

If every known vote count is zero, the script assigns a helpfulness weight of zero to those reviews to avoid division by zero.

If the helpful-vote count is missing, the helpfulness weight remains missing. Missing values are not treated as zero votes.

**Important:** this normalization depends on the dataset being processed. Adding a review with a larger helpful count can change the normalized helpfulness weights of other reviews.

### Step 12: Calculate the priority score

For reviews with known helpful-vote counts, the script combines recency and normalized helpfulness using the following configurable weights:

- Recency weight: \(w_R = 0.70\)
- Helpfulness weight: \(w_U = 0.30\)

The score is:

\[
\boxed{
P_i = 0.70R_i + 0.30U_i
}
\]

where:

- \(P_i\) is the review priority score.
- \(R_i\) is the recency weight.
- \(U_i\) is the helpfulness weight.

For reviews with a known helpful-vote count, the score generally ranges from 0 to 1.

For reviews with missing helpful-vote counts, the script uses the recency weight alone:

\[
\boxed{P_i = R_i}
\]

This fallback preserves the ability to prioritize reviews when helpfulness metadata is unavailable.

The 70% recency and 30% helpfulness weighting is a configurable starting point, not a statistically validated optimum. The weights should be evaluated against the project's analytical goals.

### Worked example

Suppose a review has these values:

- Age: 365 days
- Half-life: 730 days
- Normalized helpfulness weight: 0.60

Its recency weight is:

\[
R_i = 2^{-365/730} \approx 0.7071
\]

Its priority score is:

\[
P_i = (0.70 \times 0.7071) + (0.30 \times 0.60)
\]

\[
\boxed{P_i \approx 0.675}
\]

A review with a higher priority score is placed earlier in the ranked output.

The score should be interpreted as a relative prioritization measure within the processed dataset, not as a calibrated probability or a percentage.

### Step 13: Analyze company-response metadata separately

Developer responses are preserved but do not directly increase or decrease the priority score.

The script also records whether a response exists and calculates the response delay when the dates are valid.

\[
D_{\text{reply}} = D_{\text{response}} - D_{\text{review}}
\]

The `reply_delay_days` field contains this difference when it is non-negative. Missing or negative delays are left missing.

This information can later support analysis of company response patterns, including response coverage and time to respond.

A company reply should not automatically imply that a complaint was resolved. Understanding whether a response acknowledges an issue, provides a fix, or redirects the user to support requires a separate classification step.

---

## 6. Output Files

The script creates a `processed_reviews/` directory with the following files.

### `reviews_preprocessed.csv`

Contains one row per deduplicated review, sorted by review date with the newest reviews first.

It includes cleaned review text, review metadata, developer responses, inferred ratings, source-tracking information, and ranking features.

### `reviews_ranked.csv`

Contains the same deduplicated reviews, sorted by `priority_score` from highest to lowest. Reviews with equal priority scores are ordered using their review dates, with newer reviews first.

This file is suitable for selecting high-priority reviews for manual investigation, sentiment analysis, or issue classification.

### `parsing_issues.csv`

Records parsing issues detected by the script, including:

- Non-empty text before the first recognized entry in a file.
- Developer responses that cannot be associated with the immediately preceding user review.

This file should be inspected to identify potential extraction-format problems or developer aliases that need updating.

### `processing_summary.txt`

Reports per-file parsing counts, the number of records before and after deduplication, missing helpful-count statistics, response counts, inferred-rating counts, and ranking configuration.

---

## 7. Important Assumptions and Limitations

1. **Input format:** the parser expects author-and-date entry headers. A materially different export format may require changes to the parsing logic.

2. **Developer aliases:** the developer's displayed name must match a configured alias. Unrecognized company names may be incorrectly treated as user reviewers.

3. **Response association:** a response is attached only to the immediately preceding user review within the same source file. The script deliberately avoids assigning orphan replies to older reviews.

4. **Ratings:** ratings are inferred from filenames, not verified against individual reviews. The most-relevant export does not provide a rating through its filename.

5. **Deduplication:** only records matching normalized reviewer name, date, and review text are merged. Semantically equivalent but differently worded reviews are not automatically merged.

6. **Helpful votes:** the maximum observed count is retained across duplicate snapshots. This value may not correspond to a known snapshot date.

7. **Recency:** the current execution date affects recency scores. Historical reviews are retained rather than discarded.

8. **Missing values:** unavailable helpful counts and responses remain missing. They are not automatically interpreted as zero votes or negative evidence.

9. **Ranking weights:** the 70/30 weighting is a heuristic and has not been validated as optimal for every analysis task.

10. **No sentiment or topic classification:** this script prepares and prioritizes data. It does not itself perform sentiment analysis, extract complaint categories, identify feature requests, or determine whether an issue has been resolved.

11. **Auditability:** raw entry bodies are retained, and source information is preserved to support manual inspection and troubleshooting.

---

## 8. Recommended Next Steps

After validating the processed dataset, the next stage can focus on extracting actionable insights from the reviews.

Potential analyses include:

- **Sentiment analysis:** classify positive, negative, and neutral feedback.
- **Topic modeling and clustering:** group reviews by recurring themes such as crashes, login problems, booking errors, performance, and user-interface issues.
- **Complaint and feature-request extraction:** distinguish bug reports, usability complaints, missing capabilities, and improvement suggestions.
- **Company-response analysis:** classify responses by acknowledgment, support redirection, resolution claims, and generic replies.
- **Trend analysis:** examine how complaint topics and sentiment change over time or across star ratings.
- **Priority analysis:** combine review relevance with sentiment, issue frequency, and severity to identify problems that may deserve investigation first.

The preprocessing stage should be validated before these downstream analyses begin. In particular, manually check a representative sample of user reviews, developer replies, helpful-vote counts, ratings, and duplicate groups to ensure that the structured data accurately reflects the original exports.