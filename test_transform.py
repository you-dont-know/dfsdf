import numpy as np
import pytest
from transform import Transform3D

@pytest.fixture
def setup_transform():
    transform = Transform3D()
    # Generating random 3D points with integers between 0 and 20 for testing
    np.random.seed(42)  # For reproducibility
    point = np.random.randint(0, 20, size=3)  # Random 3D point [x, y, z]
    return transform, point

def test_translation(setup_transform):
    transform, point = setup_transform
    # Generating a random translation offset (integers between 0 and 20)
    offset = np.random.randint(0, 20, size=3)
    result = transform.translate(point, offset)
    # Expected translation
    expected = point + offset
    assert np.allclose(result, expected), f"Translation failed: {result} != {expected}"

def test_rotation_x(setup_transform):
    transform, point = setup_transform
    # Random angle for rotation around X-axis (integers between 0 and 20 degrees)
    angle = np.random.randint(0, 20)
    result = transform.rotate_x(point, angle)
    
    # Expected rotation around X-axis using numpy
    angle_radians = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                [0, np.sin(angle_radians), np.cos(angle_radians)]])
    expected = np.dot(rotation_matrix, point)
    assert np.allclose(result, expected), f"Rotation around X-axis failed: {result} != {expected}"

def test_rotation_y(setup_transform):
    transform, point = setup_transform
    # Random angle for rotation around Y-axis (integers between 0 and 20 degrees)
    angle = np.random.randint(0, 20)
    result = transform.rotate_y(point, angle)
    
    # Expected rotation around Y-axis using numpy
    angle_radians = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                [0, 1, 0],
                                [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    expected = np.dot(rotation_matrix, point)
    assert np.allclose(result, expected), f"Rotation around Y-axis failed: {result} != {expected}"

def test_rotation_z(setup_transform):
    transform, point = setup_transform
    # Random angle for rotation around Z-axis (integers between 0 and 20 degrees)
    angle = np.random.randint(0, 20)
    result = transform.rotate_z(point, angle)
    
    # Expected rotation around Z-axis using numpy
    angle_radians = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1]])
    expected = np.dot(rotation_matrix, point)
    assert np.allclose(result, expected), f"Rotation around Z-axis failed: {result} != {expected}"

def test_apply_transformation(setup_transform):
    transform, point = setup_transform
    # Random 4x4 transformation matrix with integers between 0 and 20
    transformation_matrix = np.random.randint(0, 20, size=(4, 4))
    # Enforce the last row to be [0, 0, 0, 1] for proper 3D transformations
    transformation_matrix[3] = [0, 0, 0, 1]
    
    # Apply the transformation to the point
    result = transform.apply_transformation(point, transformation_matrix)
    
    # Expected transformation using numpy
    point_homogeneous = np.append(point, 1)
    expected_homogeneous = np.dot(transformation_matrix, point_homogeneous)
    expected = expected_homogeneous[:3]
    assert np.allclose(result, expected), f"Transformation matrix application failed: {result} != {expected}"
