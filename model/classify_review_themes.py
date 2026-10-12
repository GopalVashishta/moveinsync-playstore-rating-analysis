"""Thematic Issue Classification Pipeline.

Implements Phases 1 to 4 of the Thematic Issue Classification Implementation Plan:
1. Embedding Generation: all-MiniLM-L6-v2 embeddings for 586 reviews with short-text filtering.
2. Cluster Discovery: PCA reduction + HDBSCAN clustering + cluster profiling + 2D scatter plot.
3. Theme Taxonomy + Multi-Label Assignment: Cosine similarity matching against seed taxonomy.
4. Quantification + Priority Ranking: Frequency, severity, recency, priority scores, and visualizations.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from sentence_transformers import SentenceTransformer

# Reconfigure stdout/stderr encoding for UTF-8 compatibility on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Configuration (as defined in Implementation Plan)
# -----------------------------------------------------------------------------

THEME_TAXONOMY: Dict[str, Dict[str, str]] = {
    "T1": {
        "label": "Pickup time wrong or changed",
        "seed_sentence": "The pickup time keeps changing at the last minute",
    },
    "T2": {
        "label": "GPS tracking inaccurate",
        "seed_sentence": "The live tracking shows wrong location of the cab",
    },
    "T3": {
        "label": "App crashes or won't open",
        "seed_sentence": "The app crashes every time I open it",
    },
    "T4": {
        "label": "Login or OTP failure",
        "seed_sentence": "Cannot login because OTP is not being sent",
    },
    "T5": {
        "label": "Route or routing problems",
        "seed_sentence": "The route is wrong and the cab takes a very long detour",
    },
    "T6": {
        "label": "Booking fails or forced booking",
        "seed_sentence": "Unable to book a cab, the booking keeps failing",
    },
    "T7": {
        "label": "Bad UI update or confusing interface",
        "seed_sentence": "The new update is confusing and harder to use",
    },
    "T8": {
        "label": "Driver or cab issues",
        "seed_sentence": "The cab driver was rude and did not follow the route",
    },
    "T9": {
        "label": "Notification spam or missing alerts",
        "seed_sentence": "Too many popup notifications that won't close",
    },
    "T10": {
        "label": "Customer support unresponsive",
        "seed_sentence": "Customer support does not respond or help",
    },
    "T11": {
        "label": "Slow or laggy performance",
        "seed_sentence": "The app is very slow and takes forever to load",
    },
    "T12": {
        "label": "Permissions or policy popups",
        "seed_sentence": "The app asks for too many permissions every time",
    },
}

SIMILARITY_THRESHOLD: float = 0.35
HDBSCAN_MIN_CLUSTER_SIZE: int = 10
HDBSCAN_MIN_SAMPLES: int = 5
PCA_COMPONENTS: int = 50
SHORT_TEXT_WORD_LIMIT: int = 3
PRIORITY_WEIGHTS: Dict[str, float] = {"freq": 0.4, "severity": 0.3, "recency": 0.3}

MODEL_NAME: str = "all-MiniLM-L6-v2"
MANUAL_DOWNLOAD_URL: str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def resolve_project_root() -> Path:
    """Resolve project root directory containing play_store_eda_output."""
    # Check current working directory first
    cwd = Path.cwd()
    if (cwd / "play_store_eda_output").exists():
        return cwd
    # Check parent of current file
    file_parent = Path(__file__).resolve().parent
    if (file_parent / "play_store_eda_output").exists():
        return file_parent
    if (file_parent.parent / "play_store_eda_output").exists():
        return file_parent.parent
    return cwd


# -----------------------------------------------------------------------------
# Phase 1: Embedding Generation
# -----------------------------------------------------------------------------

def load_and_embed(
    input_csv: Path,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, SentenceTransformer]:
    """
    Phase 1: Load review features and generate/cache sentence embeddings.

    Returns:
        embeddings: (N, 384) L2-normalized float32 numpy array.
        df: Enriched dataframe with `_short_text` flag.
        short_mask: Boolean mask indicating reviews with <= 3 words.
        model: Loaded SentenceTransformer instance.
    """
    logger.info("=== Phase 1: Loading reviews and preparing embeddings ===")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} reviews from {input_csv}")

    # Text cleaning & edge case: empty or NaN text
    raw_texts = df["review_text_clean"].copy()
    nan_mask = raw_texts.isna() | (raw_texts.astype(str).str.strip() == "")
    if nan_mask.any():
        logger.warning(
            f"Found {nan_mask.sum()} empty or NaN review_text_clean rows. "
            f"Replacing with 'no review text'."
        )
        raw_texts[nan_mask] = "no review text"

    texts = raw_texts.astype(str).tolist()

    # Short review flag: <= 3 words
    # Prefer existing _word_count column if present, else count words
    if "_word_count" in df.columns:
        short_mask = (df["_word_count"] <= SHORT_TEXT_WORD_LIMIT).values
    else:
        word_counts = np.array([len(t.split()) for t in texts])
        short_mask = word_counts <= SHORT_TEXT_WORD_LIMIT

    df["_short_text"] = short_mask
    logger.info(
        f"Flagged {short_mask.sum()} reviews as short text (<= {SHORT_TEXT_WORD_LIMIT} words). "
        f"Non-short reviews for clustering: {(~short_mask).sum()}"
    )

    # Check cache
    embeddings_file = output_dir / "embeddings.npy"
    ids_file = output_dir / "embedding_ids.csv"

    cache_valid = False
    if not force_recompute and embeddings_file.exists() and ids_file.exists():
        cached_ids_df = pd.read_csv(ids_file)
        if len(cached_ids_df) == len(df) and (cached_ids_df["review_id"] == df["review_id"]).all():
            cache_valid = True

    # Initialize SentenceTransformer model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(
            f"Failed to load/download sentence transformer model '{model_name}'. "
            f"Please ensure network access or download manually from {MANUAL_DOWNLOAD_URL}."
        )
        raise RuntimeError(f"Model download/initialization failed: {e}") from e

    if cache_valid:
        logger.info(f"Loading cached embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
    else:
        logger.info(f"Encoding {len(texts)} reviews with {model_name}...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_file, embeddings)
        df[["review_id", "_short_text"]].to_csv(ids_file, index=False)
        logger.info(f"Saved embeddings to {embeddings_file} and IDs to {ids_file}")

    # Validation
    expected_dim = 384
    assert embeddings.shape == (len(df), expected_dim), (
        f"Unexpected embedding shape: {embeddings.shape}, expected ({len(df)}, {expected_dim})"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not L2 normalized!"

    # Validation check: pair of pickup reviews cosine similarity
    pickup_indices = [
        i for i, t in enumerate(texts)
        if "pickup" in t.lower() or "pick up" in t.lower()
    ]
    if len(pickup_indices) >= 2:
        sim = float(np.dot(embeddings[pickup_indices[0]], embeddings[pickup_indices[1]]))
        logger.info(
            f"Validation sample similarity between pickup reviews #{pickup_indices[0]} and #{pickup_indices[1]}: {sim:.4f}"
        )

    logger.info(f"Phase 1 complete. Embedding shape: {embeddings.shape}")
    return embeddings, df, short_mask, model


# -----------------------------------------------------------------------------
# Phase 2: Cluster Discovery
# -----------------------------------------------------------------------------

def discover_clusters(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    short_mask: np.ndarray,
    output_dir: Path,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    pca_components: int = PCA_COMPONENTS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Phase 2: Perform PCA + HDBSCAN on non-short reviews, extract cluster profiles, and plot scatter.

    Returns:
        all_labels: (N,) cluster label array for all reviews (-1: noise, -2: short text).
        cluster_profiles_df: Summary profile for each discovered cluster.
    """
    logger.info("=== Phase 2: Cluster Discovery ===")
    non_short_indices = np.where(~short_mask)[0]
    non_short_embeddings = embeddings[non_short_indices]

    # PCA dimensionality reduction
    n_comp = min(pca_components, non_short_embeddings.shape[0] - 1, non_short_embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    embeddings_reduced = pca.fit_transform(non_short_embeddings)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA reduced to {n_comp} dimensions. Total explained variance: {explained_variance:.2%}")

    # Adaptive HDBSCAN clustering with edge-case handling
    curr_min_size = min_cluster_size
    curr_min_samples = min_samples

    def run_hdbscan(size: int, samples: int) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=size,
            min_samples=samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        return clusterer.fit_predict(embeddings_reduced)

    labels = run_hdbscan(curr_min_size, curr_min_samples)
    distinct_clusters = sorted([c for c in set(labels) if c != -1])
    noise_count = (labels == -1).sum()
    noise_prop = noise_count / len(labels)

    logger.info(
        f"Initial HDBSCAN (min_cluster_size={curr_min_size}, min_samples={curr_min_samples}): "
        f"{len(distinct_clusters)} clusters, noise proportion: {noise_prop:.1%}"
    )

    # Edge cases from plan:
    # 1. If < 5 clusters: reduce min_cluster_size to 8, then 6
    if len(distinct_clusters) < 5:
        for fallback_size in [8, 6]:
            logger.info(f"Clusters < 5 ({len(distinct_clusters)}). Retrying with min_cluster_size={fallback_size}")
            curr_min_size = fallback_size
            labels = run_hdbscan(curr_min_size, curr_min_samples)
            distinct_clusters = sorted([c for c in set(labels) if c != -1])
            if len(distinct_clusters) >= 5:
                break
        logger.info(f"HDBSCAN settled on min_cluster_size={curr_min_size} ({len(distinct_clusters)} clusters)")

    # 2. If > 20 clusters: increase min_cluster_size to 15, then 20
    elif len(distinct_clusters) > 20:
        for fallback_size in [15, 20]:
            logger.info(f"Clusters > 20 ({len(distinct_clusters)}). Retrying with min_cluster_size={fallback_size}")
            curr_min_size = fallback_size
            labels = run_hdbscan(curr_min_size, curr_min_samples)
            distinct_clusters = sorted([c for c in set(labels) if c != -1])
            if len(distinct_clusters) <= 20:
                break
        logger.info(f"HDBSCAN settled on min_cluster_size={curr_min_size} ({len(distinct_clusters)} clusters)")

    # 3. If noise proportion > 40%: reduce min_samples to 3
    noise_count = (labels == -1).sum()
    noise_prop = noise_count / len(labels)
    if noise_prop > 0.40:
        logger.info(f"Noise proportion {noise_prop:.1%} > 40%. Retrying with min_samples=3")
        curr_min_samples = 3
        new_labels = run_hdbscan(curr_min_size, curr_min_samples)
        new_noise_prop = (new_labels == -1).sum() / len(new_labels)
        if new_noise_prop <= noise_prop:
            labels = new_labels
            noise_prop = new_noise_prop
            distinct_clusters = sorted([c for c in set(labels) if c != -1])
        if noise_prop > 0.40:
            logger.warning(
                f"HDBSCAN noise proportion ({noise_prop:.1%}) still > 40% after reducing min_samples=3. "
                f"Proceeding as Phase 3 handles unassigned reviews."
            )

    # Assign labels back to full dataframe
    # Label -1: noise, -2: short review excluded from clustering
    all_labels = np.full(len(df), -2, dtype=int)
    all_labels[non_short_indices] = labels
    df["cluster_label"] = all_labels

    # Save cluster assignments
    assignments_file = output_dir / "cluster_assignments.csv"
    df[["review_id", "cluster_label", "_short_text"]].to_csv(assignments_file, index=False)
    logger.info(f"Saved cluster assignments to {assignments_file}")

    # Fit TF-IDF on all non-short texts for cluster profiling
    non_short_texts = df.loc[non_short_indices, "review_text_clean"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=1500,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    tfidf_matrix = tfidf.fit_transform(non_short_texts)
    feature_names = np.array(tfidf.get_feature_names_out())

    # Build cluster profiles
    profiles: List[Dict[str, Any]] = []

    for cluster_id in distinct_clusters:
        in_cluster_mask = labels == cluster_id
        cluster_sub_indices = np.where(in_cluster_mask)[0]
        cluster_orig_indices = non_short_indices[cluster_sub_indices]
        cluster_size = len(cluster_orig_indices)

        # Centroid calculation & 5 most central reviews
        cluster_emb = non_short_embeddings[cluster_sub_indices]
        centroid = cluster_emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        cos_sims = cluster_emb @ centroid
        central_order = np.argsort(cos_sims)[::-1][:5]
        central_review_ids = df.loc[cluster_orig_indices[central_order], "review_id"].tolist()
        central_snippets = [
            str(df.loc[cluster_orig_indices[i], "review_text_clean"])[:100].replace("\n", " ")
            for i in central_order
        ]

        # Top 10 TF-IDF terms
        cluster_tfidf = tfidf_matrix[cluster_sub_indices].mean(axis=0)
        cluster_tfidf_arr = np.asarray(cluster_tfidf).ravel()
        top_tfidf_idx = np.argsort(cluster_tfidf_arr)[::-1][:10]
        top_terms = feature_names[top_tfidf_idx].tolist()

        # Rating distribution
        cluster_ratings = df.loc[cluster_orig_indices, "rating"]
        r1 = int((cluster_ratings == 1.0).sum())
        r2 = int((cluster_ratings == 2.0).sum())
        r3 = int((cluster_ratings == 3.0).sum())
        r4 = int((cluster_ratings == 4.0).sum())
        r_nan = int(cluster_ratings.isna().sum())

        profiles.append({
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "pct_of_clustered": round(cluster_size / len(labels) * 100, 2),
            "top_tfidf_terms": ", ".join(top_terms),
            "central_review_ids": "; ".join(central_review_ids),
            "representative_snippet": central_snippets[0] if central_snippets else "",
            "rating_1_count": r1,
            "rating_2_count": r2,
            "rating_3_count": r3,
            "rating_4_count": r4,
            "rating_unknown_count": r_nan,
        })

    cluster_profiles_df = pd.DataFrame(profiles)
    profiles_file = output_dir / "cluster_profiles.csv"
    cluster_profiles_df.to_csv(profiles_file, index=False)
    logger.info(f"Saved cluster profiles to {profiles_file}")

    # 2D PCA Visualization
    pca_2d = PCA(n_components=2, random_state=42)
    coords_2d = pca_2d.fit_transform(embeddings)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    scatter_file = plots_dir / "cluster_scatter.png"
    plt.figure(figsize=(11, 8), dpi=150)
    cmap = matplotlib.colormaps["tab20"]

    # Plot noise points (-1) in grey
    noise_indices = np.where(all_labels == -1)[0]
    if len(noise_indices) > 0:
        plt.scatter(
            coords_2d[noise_indices, 0],
            coords_2d[noise_indices, 1],
            c="#9e9e9e",
            s=28,
            alpha=0.45,
            label=f"Noise (n={len(noise_indices)})",
            edgecolors="none",
        )

    # Plot short reviews (-2) as hollow markers
    short_indices = np.where(all_labels == -2)[0]
    if len(short_indices) > 0:
        plt.scatter(
            coords_2d[short_indices, 0],
            coords_2d[short_indices, 1],
            facecolors="none",
            edgecolors="#616161",
            s=32,
            alpha=0.6,
            marker="o",
            linewidths=1.0,
            label=f"Short reviews <=3 words (n={len(short_indices)})",
        )

    # Plot each cluster
    for idx, c in enumerate(distinct_clusters):
        c_indices = np.where(all_labels == c)[0]
        plt.scatter(
            coords_2d[c_indices, 0],
            coords_2d[c_indices, 1],
            color=cmap(idx % 20),
            s=40,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=f"Cluster {c} (n={len(c_indices)})",
        )

    plt.title(
        f"HDBSCAN Cluster Discovery (2D PCA Projection)\n"
        f"Discovered Clusters: {len(distinct_clusters)} | Total Reviews: {len(df)}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("PCA Component 1", fontsize=10)
    plt.ylabel("PCA Component 2", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(scatter_file, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cluster scatter plot to {scatter_file}")

    return all_labels, cluster_profiles_df


# -----------------------------------------------------------------------------
# Phase 3: Theme Taxonomy + Multi-Label Assignment
# -----------------------------------------------------------------------------

def assign_themes(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    model: SentenceTransformer,
    output_dir: Path,
    taxonomy: Dict[str, Dict[str, str]] = THEME_TAXONOMY,
    threshold: float = SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Phase 3: Encode seed sentences and assign themes via cosine similarity.

    Returns:
        review_themes_df: Dataframe with boolean columns for each theme,
                          plus primary_theme, max_similarity, theme_count.
    """
    logger.info("=== Phase 3: Theme Taxonomy + Multi-Label Assignment ===")
    theme_ids = list(taxonomy.keys())
    seed_sentences = [taxonomy[tid]["seed_sentence"] for tid in theme_ids]
    theme_labels = [taxonomy[tid]["label"] for tid in theme_ids]

    # Encode seed sentences with normalized embeddings
    seed_embeddings = model.encode(
        seed_sentences,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Cosine similarity matrix: N reviews x T themes
    similarity_matrix = embeddings @ seed_embeddings.T

    # Threshold check and uncategorized proportion handling
    binary_matrix = similarity_matrix >= threshold
    theme_counts = binary_matrix.sum(axis=1)
    uncategorized_count = (theme_counts == 0).sum()
    uncategorized_prop = uncategorized_count / len(df)
    logger.info(
        f"Threshold={threshold:.2f}: {uncategorized_count}/{len(df)} "
        f"({uncategorized_prop:.1%}) reviews uncategorized."
    )

    # Edge case: If > 15% uncategorized, test threshold = 0.30 as per plan
    active_threshold = threshold
    if uncategorized_prop > 0.15:
        logger.warning(
            f"Uncategorized reviews ({uncategorized_prop:.1%}) exceed 15% target at threshold {threshold}. "
            f"Lowering threshold to 0.30 per implementation plan."
        )
        test_binary = similarity_matrix >= 0.30
        test_uncat_count = int((test_binary.sum(axis=1) == 0).sum())
        test_uncat_prop = test_uncat_count / len(df)
        non_short_mask = (df["_short_text"] == False).values
        test_uncat_non_short = int(((test_binary.sum(axis=1) == 0) & non_short_mask).sum())
        test_uncat_non_short_prop = test_uncat_non_short / non_short_mask.sum()
        logger.info(
            f"Threshold=0.30 yields {test_uncat_count}/{len(df)} ({test_uncat_prop:.1%}) uncategorized overall, "
            f"and {test_uncat_non_short}/{non_short_mask.sum()} ({test_uncat_non_short_prop:.1%}) uncategorized among non-short reviews (>3 words)."
        )
        active_threshold = 0.30
        binary_matrix = test_binary
        theme_counts = binary_matrix.sum(axis=1)

    # Build review_themes DataFrame
    review_themes_data = {"review_id": df["review_id"].values}

    for idx, tid in enumerate(theme_ids):
        col_name = f"theme_{tid}_{theme_labels[idx]}"
        review_themes_data[col_name] = binary_matrix[:, idx]
        review_themes_data[f"sim_{tid}"] = np.round(similarity_matrix[:, idx], 4)

    # Primary theme, max similarity, theme count
    max_sim_idx = np.argmax(similarity_matrix, axis=1)
    max_sims = np.max(similarity_matrix, axis=1)

    primary_themes = []
    for i in range(len(df)):
        if theme_counts[i] == 0:
            primary_themes.append("Uncategorized")
        else:
            # Primary theme is the assigned theme with highest similarity
            assigned_indices = np.where(binary_matrix[i])[0]
            best_idx = assigned_indices[np.argmax(similarity_matrix[i, assigned_indices])]
            primary_themes.append(theme_labels[best_idx])

    review_themes_data["primary_theme"] = primary_themes
    review_themes_data["max_similarity"] = np.round(max_sims, 4)
    review_themes_data["theme_count"] = theme_counts
    review_themes_data["is_uncategorized"] = theme_counts == 0

    review_themes_df = pd.DataFrame(review_themes_data)
    review_themes_file = output_dir / "review_themes.csv"
    review_themes_df.to_csv(review_themes_file, index=False)
    logger.info(f"Saved review themes to {review_themes_file}")

    # Validation checks
    multi_label_count = (theme_counts > 1).sum()
    multi_label_prop = multi_label_count / len(df)
    final_uncat_prop = (theme_counts == 0).sum() / len(df)
    logger.info(
        f"Multi-label reviews: {multi_label_count} ({multi_label_prop:.1%}) | "
        f"Uncategorized: {(theme_counts == 0).sum()} ({final_uncat_prop:.1%})"
    )

    # Check if any single theme captures > 40%
    for idx, tid in enumerate(theme_ids):
        t_prop = binary_matrix[:, idx].sum() / len(df)
        if t_prop > 0.40:
            logger.warning(
                f"Theme {tid} ({theme_labels[idx]}) captures {t_prop:.1%} > 40% of reviews. "
                f"Consider refining seed sentence."
            )

    return review_themes_df


# -----------------------------------------------------------------------------
# Phase 4: Quantification + Priority Ranking
# -----------------------------------------------------------------------------

def quantify_and_rank(
    df: pd.DataFrame,
    review_themes_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    taxonomy: Dict[str, Dict[str, str]] = THEME_TAXONOMY,
    weights: Dict[str, float] = PRIORITY_WEIGHTS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 4: Calculate frequency, severity, recency, priority scores, and visualizations.

    Returns:
        theme_summary_df: Full descriptive statistics per theme.
        theme_ranking_df: Sorted priority ranking.
    """
    logger.info("=== Phase 4: Quantification + Priority Ranking ===")
    merged = pd.merge(df, review_themes_df, on="review_id")
    total_reviews = len(merged)

    theme_ids = list(taxonomy.keys())
    theme_labels = [taxonomy[tid]["label"] for tid in theme_ids]
    col_names = [f"theme_{tid}_{theme_labels[i]}" for i, tid in enumerate(theme_ids)]

    summary_rows: List[Dict[str, Any]] = []

    # Recency window: Sep–Oct 2026
    recent_months = {"2026-09", "2026-10"}

    for idx, tid in enumerate(theme_ids):
        t_label = theme_labels[idx]
        t_col = col_names[idx]
        sim_col = f"sim_{tid}"

        theme_mask = merged[t_col].values
        mentions = int(theme_mask.sum())
        freq_raw = mentions / total_reviews if total_reviews > 0 else 0.0

        if mentions == 0:
            summary_rows.append({
                "theme_id": tid,
                "theme_label": t_label,
                "mentions": 0,
                "freq_pct": 0.0,
                "severity_raw": 0.0,
                "recency_raw": 0.0,
                "rating_1_count": 0,
                "rating_2_count": 0,
                "rating_3_count": 0,
                "rating_4_count": 0,
                "rating_unknown_count": 0,
                "median_helpful_votes": 0.0,
                "helpful_signal_strong": False,
                "example_review_sim": "",
                "example_review_helpful": "",
                "example_review_recent": "",
                "trend_direction": "stable",
            })
            continue

        theme_subset = merged[theme_mask]

        # Star rating breakdown
        rated_subset = theme_subset[theme_subset["rating"].notna()]
        r1 = int((theme_subset["rating"] == 1.0).sum())
        r2 = int((theme_subset["rating"] == 2.0).sum())
        r3 = int((theme_subset["rating"] == 3.0).sum())
        r4 = int((theme_subset["rating"] == 4.0).sum())
        r_unknown = int(theme_subset["rating"].isna().sum())

        # Severity: proportion of theme's reviews that are 1★ among rated reviews
        if len(rated_subset) > 0:
            severity_raw = r1 / len(rated_subset)
        else:
            severity_raw = np.nan

        # Recency: proportion of theme's reviews in Sep–Oct 2026
        recent_count = int(theme_subset["_month"].isin(recent_months).sum())
        recency_raw = recent_count / mentions

        # Helpful votes signal
        helpful_vals = theme_subset["helpful_count"].dropna()
        median_helpful = float(helpful_vals.median()) if len(helpful_vals) > 0 else 0.0
        helpful_signal = median_helpful > 5.0

        # Top 3 example reviews:
        # 1. Highest cosine similarity
        best_sim_row = theme_subset.sort_values(by=sim_col, ascending=False).iloc[0]
        # 2. Highest helpful votes (fallback to similarity)
        best_help_row = theme_subset.sort_values(
            by=["helpful_count", sim_col],
            ascending=[False, False],
        ).iloc[0]
        # 3. Most recent (fallback to similarity)
        best_recent_row = theme_subset.sort_values(
            by=["review_date", sim_col],
            ascending=[False, False],
        ).iloc[0]

        # Monthly trend (Jun–Oct 2026)
        months_order = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
        monthly_counts = [int((theme_subset["_month"] == m).sum()) for m in months_order]
        # Check if trend is increasing (Oct > Sep or positive slope over last 3 months)
        if monthly_counts[-1] > monthly_counts[-2]:
            trend_direction = "increasing"
        elif monthly_counts[-1] < monthly_counts[-2]:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        summary_rows.append({
            "theme_id": tid,
            "theme_label": t_label,
            "mentions": mentions,
            "freq_pct": round(freq_raw * 100, 2),
            "freq_raw": freq_raw,
            "severity_raw": severity_raw,
            "recency_raw": recency_raw,
            "rating_1_count": r1,
            "rating_2_count": r2,
            "rating_3_count": r3,
            "rating_4_count": r4,
            "rating_unknown_count": r_unknown,
            "median_helpful_votes": median_helpful,
            "helpful_signal_strong": helpful_signal,
            "example_review_sim": f"{best_sim_row['review_id']}: {str(best_sim_row['review_text_clean'])[:120]}",
            "example_review_helpful": f"{best_help_row['review_id']} (votes={best_help_row['helpful_count']}): {str(best_help_row['review_text_clean'])[:120]}",
            "example_review_recent": f"{best_recent_row['review_id']} ({best_recent_row['review_date']}): {str(best_recent_row['review_text_clean'])[:120]}",
            "trend_direction": trend_direction,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Min-max scaling for priority score components
    def min_max_scale(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) == 0 or valid.max() == valid.min():
            return pd.Series(0.5, index=series.index)
        return (series - valid.min()) / (valid.max() - valid.min())

    summary_df["freq_norm"] = min_max_scale(summary_df["freq_raw"])
    summary_df["severity_norm"] = min_max_scale(summary_df["severity_raw"].fillna(0))
    summary_df["recency_norm"] = min_max_scale(summary_df["recency_raw"])

    # Compute Priority Score: P_t = 0.4*freq + 0.3*severity + 0.3*recency
    w_f = weights.get("freq", 0.4)
    w_s = weights.get("severity", 0.3)
    w_r = weights.get("recency", 0.3)

    priority_scores = []
    for _, row in summary_df.iterrows():
        if pd.isna(row["severity_raw"]):
            # Missing rating edge case from plan: w_f = 0.55, w_r = 0.45
            p = 0.55 * row["freq_norm"] + 0.45 * row["recency_norm"]
        else:
            p = w_f * row["freq_norm"] + w_s * row["severity_norm"] + w_r * row["recency_norm"]
        priority_scores.append(round(p, 4))

    summary_df["priority_score"] = priority_scores

    # Save theme_summary.csv
    summary_file = output_dir / "theme_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved theme summary to {summary_file}")

    # Generate theme_priority_ranking.csv sorted by priority_score descending
    ranking_cols = [
        "theme_id",
        "theme_label",
        "priority_score",
        "mentions",
        "freq_pct",
        "severity_raw",
        "recency_raw",
        "trend_direction",
        "median_helpful_votes",
        "helpful_signal_strong",
        "example_review_sim",
    ]
    ranking_df = summary_df.sort_values(by="priority_score", ascending=False)[ranking_cols].reset_index(drop=True)
    ranking_file = output_dir / "theme_priority_ranking.csv"
    ranking_df.to_csv(ranking_file, index=False)
    logger.info(f"Saved theme priority ranking to {ranking_file}")

    # Generate Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Theme Frequency Bar Chart
    freq_sorted = summary_df.sort_values(by="mentions", ascending=True)
    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.barh(
        freq_sorted["theme_label"],
        freq_sorted["mentions"],
        color="#2b5c8f",
        edgecolor="#1a365d",
    )
    for bar in bars:
        width = bar.get_width()
        pct = (width / total_reviews) * 100
        plt.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)} ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
    plt.title("Theme Mention Frequency Across Reviews", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Review Mentions Count", fontsize=10)
    plt.xlim(0, max(freq_sorted["mentions"]) * 1.15)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plots_dir / "theme_frequency_bar.png", bbox_inches="tight")
    plt.close()

    # 2. Theme x Rating Heatmap
    heatmap_data = []
    heatmap_labels = []
    for _, row in summary_df.iterrows():
        heatmap_labels.append(row["theme_label"])
        heatmap_data.append([
            row["rating_1_count"],
            row["rating_2_count"],
            row["rating_3_count"],
            row["rating_4_count"],
        ])

    heatmap_arr = np.array(heatmap_data, dtype=float)
    # Normalize per theme to get rating mix percentages
    row_sums = heatmap_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    heatmap_pct = (heatmap_arr / row_sums) * 100

    plt.figure(figsize=(9, 7), dpi=150)
    im = plt.imshow(heatmap_pct, cmap="YlOrRd", aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("% of Theme Reviews", rotation=270, labelpad=15)

    plt.xticks(np.arange(4), ["1-Star", "2-Star", "3-Star", "4-Star"], fontsize=9.5)
    plt.yticks(np.arange(len(heatmap_labels)), heatmap_labels, fontsize=9)

    for i in range(len(heatmap_labels)):
        for j in range(4):
            val = heatmap_pct[i, j]
            count = int(heatmap_arr[i, j])
            text_color = "white" if val > 55 else "black"
            plt.text(j, i, f"{val:.0f}%\n({count})", ha="center", va="center", color=text_color, fontsize=8)

    plt.title("Theme Rating Composition (% and Count per Star Rating)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "theme_rating_heatmap.png", bbox_inches="tight")
    plt.close()

    # 3. Monthly Trend Line Chart (top 5 themes by mentions)
    top_5_themes = summary_df.sort_values(by="mentions", ascending=False).head(5)
    months = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]

    plt.figure(figsize=(10, 5.5), dpi=150)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (_, row) in enumerate(top_5_themes.iterrows()):
        t_col = f"theme_{row['theme_id']}_{row['theme_label']}"
        theme_sub = merged[merged[t_col]]
        counts = [int((theme_sub["_month"] == m).sum()) for m in months]
        plt.plot(
            range(len(months)),
            counts,
            marker="o",
            linewidth=2,
            color=palette[idx % len(palette)],
            label=f"{row['theme_label']} (total={row['mentions']})",
        )

    plt.xticks(range(len(months)), months, fontsize=9.5)
    plt.title("Monthly Mention Trend for Top 5 Themes (Jun–Oct 2026)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Month", fontsize=10)
    plt.ylabel("Mentions Count", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "theme_monthly_trend.png", bbox_inches="tight")
    plt.close()

    logger.info("Saved all visualization plots to theme_analysis/plots/")
    return summary_df, ranking_df


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main(
    input_csv_path: Optional[str] = None,
    output_dir_path: Optional[str] = None,
    force_recompute_embeddings: bool = False,
) -> None:
    """Execute the full classification pipeline."""
    project_root = resolve_project_root()
    logger.info(f"Project root resolved to: {project_root}")

    if input_csv_path:
        input_csv = Path(input_csv_path)
    else:
        input_csv = project_root / "play_store_eda_output" / "review_level_eda_features.csv"

    if output_dir_path:
        output_dir = Path(output_dir_path)
    else:
        output_dir = project_root / "theme_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load and Embed
    embeddings, df, short_mask, model = load_and_embed(
        input_csv=input_csv,
        output_dir=output_dir,
        model_name=MODEL_NAME,
        force_recompute=force_recompute_embeddings,
    )

    # Phase 2: Cluster Discovery
    cluster_labels, cluster_profiles_df = discover_clusters(
        embeddings=embeddings,
        df=df,
        short_mask=short_mask,
        output_dir=output_dir,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        pca_components=PCA_COMPONENTS,
    )

    # Phase 3: Theme Taxonomy + Multi-Label Assignment
    review_themes_df = assign_themes(
        embeddings=embeddings,
        df=df,
        model=model,
        output_dir=output_dir,
        taxonomy=THEME_TAXONOMY,
        threshold=SIMILARITY_THRESHOLD,
    )

    # Phase 4: Quantification + Priority Ranking
    theme_summary_df, theme_ranking_df = quantify_and_rank(
        df=df,
        review_themes_df=review_themes_df,
        embeddings=embeddings,
        output_dir=output_dir,
        taxonomy=THEME_TAXONOMY,
        weights=PRIORITY_WEIGHTS,
    )

    print("\n" + "=" * 60)
    print("THEME PRIORITY RANKING SUMMARY")
    print("=" * 60)
    for idx, row in theme_ranking_df.iterrows():
        print(
            f"{idx+1}. [{row['theme_id']}] {row['theme_label']}: "
            f"Score={row['priority_score']:.3f} | Mentions={row['mentions']} ({row['freq_pct']}%) | "
            f"Trend={row['trend_direction']}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
