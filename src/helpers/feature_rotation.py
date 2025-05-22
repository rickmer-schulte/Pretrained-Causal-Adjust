# helpers/feature_rotation.py

import numpy as np

def create_block_diagonal_rotation_matrix(feature_dim, dimensions, angles_degrees):
    """
    Create a block-diagonal rotation matrix for specified dimension pairs with varying rotation angles.

    Parameters
    ----------
    feature_dim : int
        Total number of feature dimensions.
    dimensions : list of tuples
        Each tuple contains two integers representing the dimension indices to rotate.
    angles_degrees : list of floats
        Rotation angles in degrees for each dimension pair.

    Returns
    -------
    rotation_matrix : np.ndarray
        The resulting block-diagonal rotation matrix of shape (feature_dim, feature_dim).
    """
    rotation_matrix = np.identity(feature_dim)

    for (dim1, dim2), angle_degrees in zip(dimensions, angles_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)

        # Apply rotation to the block corresponding to the dimension pair
        rotation_matrix[dim1, dim1] = cos_theta
        rotation_matrix[dim1, dim2] = -sin_theta
        rotation_matrix[dim2, dim1] = sin_theta
        rotation_matrix[dim2, dim2] = cos_theta

    return rotation_matrix

def rotate_latent_features(all_features, angle_degrees=None, num_pairs=100):
    """
    Rotate high-dimensional latent features by applying 2D rotations within multiple subspaces.

    Parameters
    ----------
    all_features : np.ndarray
        Original high-dimensional latent feature vectors of shape (N_samples, feature_dim).
    angle_degrees : float or None
        The rotation angle in degrees. If None, a random angle is chosen for each dimension pair.
    num_pairs : int
        Number of dimension pairs to rotate. Can exceed feature_dim/2, with replacement used for sampling in that case.

    Returns
    -------
    rotated_features : np.ndarray
        Rotated high-dimensional latent feature vectors of shape (N_samples, feature_dim).
    """
    feature_dim = all_features.shape[1]

    # Determine if sampling with or without replacement
    replace = 2 * num_pairs > feature_dim

    # Select dimension pairs
    dims = np.random.choice(feature_dim, size=2 * num_pairs, replace=replace)
    dimension_pairs = list(zip(dims[::2], dims[1::2]))

    # Generate random angles if not specified
    if angle_degrees is None:
        angles_degrees = np.random.uniform(0, 360, size=num_pairs)
    else:
        angles_degrees = [angle_degrees] * num_pairs

    # Create the rotation matrix
    rotation_matrix = create_block_diagonal_rotation_matrix(feature_dim, dimension_pairs, angles_degrees)

    # Apply rotation
    rotated_features = all_features @ rotation_matrix.T 

    return rotated_features