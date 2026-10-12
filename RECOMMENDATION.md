# MoveInSync Play Store Rating Analysis: Findings & Recommendations

## 1. Current State

| Metric | Value | Source |
|---|---|---|
| Play Store Rating | **4.3 ★** out of 5 | Google Play Store (as of Oct 2026) |
| Total Play Store Reviews | ~114,000+ | AppBrain / Google Play |
| Downloads | 1,000,000+ | Google Play |
| Daily Active Users (company-stated) | 1,500,000+ | MoveInSync company profile |
| Enterprise Customers | 430+ | MoveInSync company profile |

### Analysis Sample

This analysis is based on **586 unique reviews** manually collected from the Play Store between April and October 2026:

- **292** 1★ reviews (newest, Jun–Oct 2026)
- **40** 2★ reviews (newest, Apr–Oct 2026)
- **42** 3★ reviews (newest, Apr–Oct 2026)
- **48** 4★ reviews (newest, Jun–Oct 2026)
- **164** reviews from the Play Store "Most Relevant" filter (no star rating metadata; spans 2018–2026)

5-star reviews were not collected — they were overwhelmingly generic positive ("good app", "nice") and did not contain actionable feedback.

---

## 2. Why the Ratings Are Lower Than They Should Be

Semantic analysis of all 586 reviews using sentence-transformer embeddings (`all-MiniLM-L6-v2`) and cosine-similarity classification identified **12 recurring complaint themes**. The themes below are ranked by a composite priority score weighing frequency (40%), 1★ severity (30%), and recency (30%).

### Tier 1 — Immediate: User-Blocking Failures

These issues prevent users from completing basic tasks. They generate the most intense negative sentiment and are likely driving retaliatory 1★ ratings.

---

#### 🔴 Login & OTP Failure (T4) — Priority Score: 0.45

| Metric | Value |
|---|---|
| Reviews mentioning this | 54 (9.2%) |
| 1★ severity | **87.5%** (28 of 32 rated reviews) |
| Sep–Oct 2026 recency | 22.2% |
| Trend | Decreasing (but still active) |

**What users say:**

> *"Didn't get any OTP to login"* — R_e0fcd510a9587f70

> *"After the recent update, whenever I try to login for checking the cab details the text box entering the OTP doesn't appear."* — R_4f48f7c294012114 (11 helpful votes)

> *"App is not sending OTP"* — R_8d167da570bcf9e4 (Oct 6, 2026)

**Root cause observed:** OTP SMS delivery failures and a broken OTP text-input field on recent app builds. Users cannot enter the app at all — this turns the app into a brick on trip mornings.

---

#### 🔴 Customer Support Unreachable (T10) — Priority Score: 0.44

| Metric | Value |
|---|---|
| Reviews mentioning this | 69 (11.8%) |
| 1★ severity | **91.5%** (43 of 47 rated reviews — highest in the dataset) |
| Sep–Oct 2026 recency | 14.5% |
| Trend | Decreasing |

**What users say:**

> *"No option to contact help desk"* — R_8fe74cd3e1437f76

> *"The cab was shown at a different location due to a glitch in the app and support refused to help or reassign."* — R_324447462fa4ce5f (15 helpful votes)

> *"Who the hell is giving them 5 star rating, worst punctuation, my cab has to assign on 2:30 AM and it's [still not here]"* — R_6b5cf977057331fe (Oct 1, 2026)

**Root cause observed:** In-app support channel is either absent or fails silently. Users who encounter any other issue (late cab, GPS freeze, booking failure) have no way to get help in real time, which escalates frustration into a Play Store review.

---

### Tier 2 — Near-Term: Core App Stability

These issues affect every session. They are the highest-volume complaint categories and erode trust in the app's basic reliability.

---

#### 🟠 App Slowness & Lag (T11) — Priority Score: 0.55 (highest)

| Metric | Value |
|---|---|
| Reviews mentioning this | **233 (39.8%)** — most mentioned theme |
| 1★ severity | 72.2% |
| Sep–Oct 2026 recency | 18.5% |
| Median helpful votes | 3.0 |
| Trend | Decreasing |

**What users say:**

> *"This app very slow"* — R_2c312acb00db31c9

> *"The only purpose of this app is to help real time tracking of the cab, which never happens, it always gets stuck at some place while the cab has already arrived or left."* — R_61a507cf377f9f0f (70 helpful votes)

**Root cause observed:** Heavy main-thread work, likely from unoptimized map rendering and high-frequency geolocation stream updates. The perceived "slowness" also includes frozen GPS state (see T2).

---

#### 🟠 App Crashes & Update-Induced Failures (T3) — Priority Score: 0.47

| Metric | Value |
|---|---|
| Reviews mentioning this | 187 (31.9%) |
| 1★ severity | 72.7% |
| Sep–Oct 2026 recency | 18.2% |
| Median helpful votes | 3.0 |
| Trend | Decreasing |

**What users say:**

> *"Whenever I update this app, I am not able to open the app again, I need to uninstall and reinstall again. Facing this issue since 3 months."* — R_dc016e5b5788f671 (Oct 9, 2026)

> *"Every now and then I receive notifications saying that I have an older version in my mobile and reminds me to update it, but when I open it fails..."* — R_b5a3c409f2f5d539 (69 helpful votes)

**Root cause observed:** Corrupted cached session state or migration failures following app updates. Users are forced into a reinstall loop. The update-nag notifications compound frustration.

---

#### 🟠 Permissions & Policy Popups (T12) — Priority Score: 0.44

| Metric | Value |
|---|---|
| Reviews mentioning this | 170 (29.0%) |
| 1★ severity | 73.0% |
| Sep–Oct 2026 recency | 17.6% |
| Median helpful votes | 2.0 |
| Trend | Decreasing |

**What users say:**

> *"It asks too much permission when I open the app on regular basis"* — R_06422efecc2e685b

> *"For every button click it is asking for policy acceptance, which is very unprofessional."* — R_39be46a425196208 (Oct 5, 2026)

**Root cause observed:** Permission and policy acceptance dialogs re-trigger every session or on every action, suggesting the consent state is not being persisted correctly.

---

### Tier 3 — Medium-Term: Commute Operations & Real-Time Accuracy

These are the product's core value proposition. Users accept some operational variability, but broken real-time data destroys the trust that the app is useful at all.

---

#### 🟡 Route & Routing Problems (T5) — Priority Score: 0.42

| Metric | Value |
|---|---|
| Reviews mentioning this | 159 (27.1%) |
| 1★ severity | 77.5% |
| Sep–Oct 2026 recency | 12.6% |
| Median helpful votes | 3.0 |

> *"MoveOutSync doesn't plan commutes; it plans cross-country pilgrimages. The ETA is an abstract joke where a 1-hour delay is considered punctual."* — R_c8e4690778dca1aa (Oct 9, 2026)

---

#### 🟡 GPS Tracking Inaccurate (T2) — Priority Score: 0.35

| Metric | Value |
|---|---|
| Reviews mentioning this | 161 (27.5%) |
| 1★ severity | 72.6% |
| Sep–Oct 2026 recency | 11.2% |
| Median helpful votes | **4.0** (highest agreement) |

> *"Not able to see real time tracking of my rostered cab. After updating the newer version, not able to track the cab."* — R_9715069da52fa252

---

#### 🟡 Pickup Time Wrong or Changed (T1) — Priority Score: 0.36

| Metric | Value |
|---|---|
| Reviews mentioning this | 152 (25.9%) |
| 1★ severity | 75.7% |
| Sep–Oct 2026 recency | 10.5% |
| Median helpful votes | 3.0 |

> *"Driver arrival shows 5 mins, when we leave home to the pick point, changes to 30 mins. I mean changing continuously."* — R_3c6912408d53122e

---

### Additional Issues

| Theme | Mentions | 1★ Severity | Priority | Key Signal |
|---|---|---|---|---|
| T6: Booking fails or forced booking | 155 (26.5%) | 69.2% | 0.31 | Forced auto-booking without user consent |
| T8: Driver or cab issues | 110 (18.8%) | 74.2% | 0.30 | Driver no-shows, unreachable numbers |
| T9: Notification spam | 13 (2.2%) | 66.7% | 0.30 | Undismissable popups blocking app use |
| T7: Bad UI update | 81 (13.8%) | 68.9% | 0.25 | Users want the old simpler version back |

A notable micro-cluster was also discovered via HDBSCAN: **forced post-trip rating with no skip option** (7 reviews). Users are being blocked from using the app unless they rate their ride, which ironically drives retaliatory 1★ Play Store reviews.

---

## 3. Prioritized Action Plan

### P0 — Fix This Week (User-Blocking)

| # | Action | Owner | Evidence |
|---|---|---|---|
| 1 | **Audit the OTP login flow on latest app builds.** The OTP text-input field is not rendering for some users post-update. Provide fallback verification (email OTP, WhatsApp OTP, or call-in code). | Mobile App Team | T4: 87.5% 1★ severity, active Oct 2026 |
| 2 | **Add a persistent, visible support contact on the active trip screen** (chat button, phone number, or SOS). Current in-app support is either hidden or broken. Users with time-sensitive commute problems have no escalation path. | Product + Support Ops | T10: 91.5% 1★ severity, highest in dataset |
| 3 | **Add a "Skip" button to the post-trip rating prompt.** Users who are forced to rate with no skip option are going to the Play Store to give 1★ out of spite. | Product UX | HDBSCAN Cluster 0 (7 reviews, verbatim: *"forcing me to give rating after every ride, so I came here to give you 1 star, because if I don't rate I can't use the app as there is no skip option"*) |

### P1 — Fix This Sprint (Core Stability)

| # | Action | Owner | Evidence |
|---|---|---|---|
| 4 | **Fix the update-crash loop.** Implement clean session state migration on app startup so updates don't require uninstall/reinstall. Suppress update-nag notifications until the migration is verified. | Android Core Team | T3: 31.9% of reviews, 72.7% 1★, active Oct 2026 |
| 5 | **Profile and optimize the map/tracking view.** Debounce geolocation stream updates, move heavy computation off the main thread. The map view is the most-used screen and the primary source of "slow app" complaints. | Android Core Team | T11: 39.8% of reviews (most frequent), 72.2% 1★ |
| 6 | **Persist permission and policy consent state.** Users should not be re-prompted for permissions or policy acceptance every session. Investigate whether `SharedPreferences` / secure storage is being cleared on update. | Android Core Team | T12: 29.0% of reviews, 73.0% 1★ |

### P2 — Fix This Quarter (Operations & Routing)

| # | Action | Owner | Evidence |
|---|---|---|---|
| 7 | **Implement a driver GPS heartbeat monitor.** When driver telemetry drops or freezes, notify the rider explicitly ("Driver location unavailable") instead of showing a stale position. | Routing & Fleet Backend | T2: 27.5% of reviews, highest helpful-vote median (4.0) |
| 8 | **Audit ETA calculation.** Users report ETA ≠ Google Maps ETA by a factor of 3×. Compare the routing engine's estimates against Google Maps API for the same origin-destination pairs. | Routing Backend | T1: 25.9% of reviews, 75.7% 1★ |
| 9 | **Add explicit pickup-time-change notifications.** When pickup time shifts by >5 minutes, push a notification with the old time, new time, and reason. Don't silently update the UI. | Product + Fleet Backend | T1: *"Driver arrival shows 5 mins... changes to 30 mins"* |
| 10 | **Review route-clubbing logic.** Multiple users report being clubbed with riders in distant locations, creating excessive detours. Tighten the radius for co-rider clubbing. | Routing Backend | T5: 27.1% of reviews, 77.5% 1★ |

---

## 4. Methodology

### Data Collection
- **Source:** Google Play Store, MoveInSync app
- **Method:** Manual copy of reviews by star rating (1–4★ newest, Apr/Jun–Oct 2026) and Play Store "Most Relevant" filter
- **Deliberate exclusion:** 5-star reviews were inspected and excluded — they were predominantly non-informational generic praise ("good", "nice app") and did not contribute actionable insights

### Pipeline
```
Raw .txt files (5 files, 670 entries)
  → preprocess_play_store_reviews.py  →  586 unique reviews (84 duplicates merged)
  → play_store_eda.py                 →  20 CSVs, 19 plots, statistical profile
  → classify_review_themes.py         →  Embeddings, clusters, 12-theme taxonomy, priority ranking
```

### Thematic Classification Method
1. **Embedding:** All 586 review texts encoded with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional normalized vectors)
2. **Cluster discovery:** PCA (50 components, 74.3% variance) → HDBSCAN (min_cluster_size=6, min_samples=3). 82 short-text reviews (≤3 words) excluded from clustering but included in classification
3. **Multi-label assignment:** Cosine similarity between each review embedding and 12 theme seed-sentence embeddings. Threshold: 0.30. Reviews can receive multiple theme labels
4. **Priority scoring:** $P_t = 0.4 \cdot \text{freq}_t + 0.3 \cdot \text{severity}_t + 0.3 \cdot \text{recency}_t$ with min-max normalization per component

### Tools
- Python 3.13, pandas, matplotlib, numpy
- sentence-transformers 6.1.0, scikit-learn, hdbscan
- All analysis runs locally — no external API calls

### Limitations
- **Not a population sample.** This is a convenience sample of 586 reviews from a pool of ~114K. Frequencies indicate theme prevalence *within the sample*, not among all users.
- **No 5★ reviews.** We cannot identify what users like or compute an estimated overall rating from this sample.
- **Collection period bias.** 77.0% of reviews are from Apr–Oct 2026 (the active collection window). Older reviews come only from "Most Relevant" and are not representative of historical volume.
- **Non-English reviews** (Hindi transliteration, Tagalog) are a small minority and may be misclassified by the English-trained embedding model.
- **Multi-label overlap.** 67% of reviews received 2+ theme tags. This inflates individual theme counts — themes should not be summed to a total.

---

## 5. Monitoring: "How Are Our Play Store Ratings?"

The assignment states this question will be asked repeatedly. The pipeline is designed to be re-run:

### To Re-Run the Analysis

1. Collect fresh reviews from the Play Store (same method: newest 1–4★ + most relevant)
2. Place `.txt` files in `reviews/`
3. Run:
   ```
   python preprocess_play_store_reviews.py
   python play_store_eda.py
   python classify_review_themes.py
   ```
4. Compare `theme_analysis/theme_priority_ranking.csv` against the previous run

### What to Track Over Time

| Metric | How to Measure | Target |
|---|---|---|
| Overall Play Store rating | Check Play Store page | Move from 4.3 → 4.5+ |
| T4 (Login/OTP) mention rate | `theme_priority_ranking.csv` freq_pct | Drop below 5% |
| T10 (Support) 1★ severity | `theme_priority_ranking.csv` severity_raw | Drop below 50% |
| T11 (Slowness) mention rate | `theme_priority_ranking.csv` freq_pct | Drop below 20% |
| T3 (Crashes) mention rate | `theme_priority_ranking.csv` freq_pct | Drop below 15% |
| New theme emergence | Compare theme list across runs | Detect new failure modes early |
