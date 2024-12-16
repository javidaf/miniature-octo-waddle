import numpy as np
from astroglial_analysis.utils import get_formated_region_coords


def extract_features(masks, classifications):
    """
    Extracts features from masks and classifications, including:
    - Area
    - Centroid (x, y)
    - Bounding box width and height
    - Eccentricity (shape measure)
    - Directions of the first and second principal components (PC1 and PC2)
    - Magnitude of the center-of-mass shift from the geometric center

    Args:
        masks (np.ndarray): A 2D array where each pixel is labeled with a cell ID.
        classifications (list of tuples): Each tuple (class_label, cell_label)
            class_label: The type of the cell.
            cell_label: The ID of the cell in the mask.

    Returns:
        features (np.ndarray): A 2D array of shape (n_cells, n_features).
        targets (np.ndarray): A 1D array of shape (n_cells,) representing class labels.
    """
    features = []
    targets = []

    for class_label, cell_label in classifications:
        mask = np.where(masks == cell_label)
        region_coords = get_formated_region_coords(mask)

        if len(region_coords) == 0:
            continue

        area = len(region_coords)

        centroid = np.mean(region_coords, axis=0)  # [mean_x, mean_y]

        # Compute bounding box
        min_xy = np.min(region_coords, axis=0)
        max_xy = np.max(region_coords, axis=0)
        width = max_xy[0] - min_xy[0] + 1
        height = max_xy[1] - min_xy[1] + 1

        geometric_center = np.array(
            [(min_xy[0] + max_xy[0]) / 2.0, (min_xy[1] + max_xy[1]) / 2.0]
        )

        # Compute the magnitude of the shift between centroid and geometric center
        center_of_mass_shift_vector = centroid - geometric_center
        center_of_mass_shift_magnitude = np.linalg.norm(center_of_mass_shift_vector)

        # Compute covariance matrix and eigen decomposition for PCA
        if area > 1:
            cov = np.cov(region_coords, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eig(cov)

            # Sort eigenvalues and eigenvectors in descending order of eigenvalues
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Principal component directions (unit vectors)
            pc1_dir = eigenvecs[:, 0] / np.linalg.norm(eigenvecs[:, 0])
            pc2_dir = eigenvecs[:, 1] / np.linalg.norm(eigenvecs[:, 1])

            if eigenvals[0] <= 0:
                eccentricity = 0.0
            else:
                major_axis_length = 4.0 * np.sqrt(eigenvals[0])
                minor_axis_length = 4.0 * np.sqrt(eigenvals[-1])
                eccentricity = np.sqrt(
                    1 - (minor_axis_length**2 / major_axis_length**2)
                )
        else:
            # If the cell has only one pixel, skip PCA computations
            eccentricity = 0.0
            pc1_dir = np.array([1.0, 0.0])
            pc2_dir = np.array([0.0, 1.0])
        # Collect features for this cell
        cell_features = [
            area,
            centroid[0],
            centroid[1],
            width,
            height,
            eccentricity,
            pc1_dir[0],
            pc1_dir[1],
            pc2_dir[0],
            pc2_dir[1],
            center_of_mass_shift_magnitude,
        ]

        features.append(cell_features)
        targets.append(class_label)

    features = np.array(features)
    targets = np.array(targets)
    return features, targets
