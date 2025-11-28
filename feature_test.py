import os
import numpy as np
from sklearn.cluster import KMeans


CIRCLES_DIR = "facebook"   # folder with ego feature files


def load_features_for_ego(ego_id, base_dir="facebook"):
    feat_path = os.path.join(base_dir, f"{ego_id}.feat")
    featnames_path = os.path.join(base_dir, f"{ego_id}.featnames")
    egofeat_path = os.path.join(base_dir, f"{ego_id}.egofeat")

    # Load feature names
    featnames = []
    with open(featnames_path, "r") as f:
        for line in f:
            featnames.append(line.strip())

    # Load alter feature vectors
    alters = []
    alter_ids = []
    with open(feat_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            node = int(parts[0])
            feats = list(map(int, parts[1:]))
            alter_ids.append(node)
            alters.append(feats)

    # Convert to numpy matrix
    X = np.array(alters)

    return featnames, alter_ids, X


def cluster_features(ego_id, k=5):
    print(f"\n===== FEATURE-BASED CLUSTERING FOR EGO {ego_id} =====")

    featnames, alter_ids, X = load_features_for_ego(ego_id)

    print(f"Number of alters: {len(alter_ids)}")
    print(f"Number of features per alter: {X.shape[1]}")

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    # Count cluster sizes
    cluster_sizes = {i: 0 for i in range(k)}
    for lbl in labels:
        cluster_sizes[lbl] += 1

    print("\nCluster sizes:")
    for lbl, size in cluster_sizes.items():
        print(f"  Cluster {lbl}: {size} users")

    # Inspect one example from each cluster
    print("\nExample members from each cluster:")
    for lbl in range(k):
        members = [alter_ids[i] for i in range(len(labels)) if labels[i] == lbl]
        print(f"  Cluster {lbl}: sample members {members[:10]} ...")

    # Return labels if needed for plotting later
    return labels, alter_ids


def main():
    # Try clustering features for ego 107
    cluster_features("107", k=5)


if __name__ == "__main__":
    main()
