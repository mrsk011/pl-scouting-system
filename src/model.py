from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_PATH = Path("data/processed/players_per90.csv")

FEATURE_COLS = [
    "G-PK_per90",
    "Ast_per90",
    "npxG_per90",
    "xAG_per90",
    "PrgC_per90",
    "PrgP_per90",
    "PrgR_per90",
    "CrdY_per90",
]

CLUSTER_NAMES = {
    0: "Creative Wingers & Forwards",
    1: "Defenders & Holding Midfielders",
    2: "Classic Strikers and Poachers",
    3: "Playmakers & Progressive Passers",
    4: "Attacking Full-Backs & Wing-Backs",
}


def evaluate_k_range(df: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    # Compute inertia (Elbow) and silhouette score across k values.
    X_scaled = StandardScaler().fit_transform(df[FEATURE_COLS])
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": round(km.inertia_, 2),
                "silhouette": round(silhouette_score(X_scaled, labels), 3),
            }
        )
    return pd.DataFrame(rows)


def build_scouting_model(data_path: Path = PROCESSED_DATA_PATH,
    n_clusters: int = 5,) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    #Standardize features, fit K-Means and PCA, and compute Cosine Similarity.
    df = pd.read_csv(data_path).copy()

    # 1. Z-score standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])

    # 2. K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    df["Playing_Style"] = df["Cluster"].map(
        lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}")
    )

    # 3. 2D PCA projection for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["PC1"] = coords[:, 0].round(3)
    df["PC2"] = coords[:, 1].round(3)

    # 4. Percentile ranks (0-100) for comparison charts
    for col in FEATURE_COLS:
        df[f"{col}_pct"] = (df[col].rank(pct=True) * 100).round(1)

    # 5. Pairwise Cosine Similarity matrix
    sim_matrix = cosine_similarity(X_scaled)

    return df, sim_matrix, pca


def get_similar_players(
    player_name: str,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    top_n: int = 5,
    position_filter: list[str] | None = None,
    age_range: tuple[int, int] | None = None,
    min_90s: float = 0.0,
) -> pd.DataFrame:
    #Return top_n most statistically similar players with optional filters.
    matches = df.index[df["Player"] == player_name].tolist()
    if not matches:
        raise ValueError(f"Player '{player_name}' not found in dataset.")

    idx = matches[0]
    scores = pd.Series(sim_matrix[idx], index=df.index, name="Similarity")

    candidates = df.copy()
    candidates["Similarity (%)"] = (scores * 100).round(1)

    # Exclude the searched player
    candidates = candidates.drop(index=idx)

    # Apply scouting filters
    if position_filter:
        candidates = candidates[candidates["Primary_Pos"].isin(position_filter)]
    if age_range:
        candidates = candidates[
            (candidates["Age"] >= age_range[0]) & (candidates["Age"] <= age_range[1])
        ]
    if min_90s > 0:
        candidates = candidates[candidates["90s"] >= min_90s]

    cols = ["Player", "Squad", "Primary_Pos", "Age", "90s", "Playing_Style", "Similarity (%)"]
    return (
        candidates.sort_values("Similarity (%)", ascending=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )

if __name__ == "__main__":
    model_df, sim_matrix, pca = build_scouting_model()

    print("PCA Explained Variance:", [round(v * 100, 1) for v in pca.explained_variance_ratio_])
    print("\nCluster Distribution:")
    print(model_df["Playing_Style"].value_counts())

    print("\nTop 5 statistical matches for Mohamed Salah:")
    print(get_similar_players("Mohamed Salah", model_df, sim_matrix, top_n=5))

    print("\nTop 5 U-24 matches for Mohamed Salah:")
    print(get_similar_players("Mohamed Salah", model_df, sim_matrix, top_n=5, age_range=(16, 23)))