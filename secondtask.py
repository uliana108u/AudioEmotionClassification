import numpy as np

def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    matches = []

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return matches

    # 1. Compute pairwise distances (Euclidean)
    dist_matrix = np.sqrt(np.sum((des1[:, np.newaxis] - des2) ** 2, axis=2))

    # 2. Find nearest neighbors in both directions
    # For each descriptor in des1, find the closest in des2
    closest_des2_for_des1 = np.argmin(dist_matrix, axis=1)
    min_dist_des1_to_des2 = np.min(dist_matrix, axis=1)

    # For each descriptor in des2, find the closest in des1
    closest_des1_for_des2 = np.argmin(dist_matrix, axis=0)
    min_dist_des2_to_des1 = np.min(dist_matrix, axis=0)

    # 3. Cross-check: keep only mutual best matches
    for i in range(len(des1)):
        j = closest_des2_for_des1[i]
        if closest_des1_for_des2[j] == i:  # Check if mutual
            distance = min_dist_des1_to_des2[i]
            matches.append(DummyMatch(i, j, distance))

    # 4. Sort matches by distance (ascending)
    matches.sort(key=lambda m: m.distance)

    return matches

