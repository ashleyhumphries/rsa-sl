import numpy as np


#checking bold data file loading

expected_bold_shape = (65, 77, 56, 180)
expected_mask_shape = (65, 77, 56)

def check_bold_data_shape(bold_data, expected_shape):
    """
    Check if the shape of BOLD data matches the expected shape.
    """
    assert bold_data.shape == expected_bold_shape, f"Unexpected BOLD data shape: {bold_data.shape}. Expected shape: {expected_bold_shape}"
    print(f"BOLD data shape: {bold_data.shape} - OK")

def check_mask_shape(mask_data, expected_shape):
    """
    Check if the shape of the mask data matches the expected shape.
    """
    assert mask_data.shape == expected_mask_shape, f"Unexpected mask shape: {mask_data.shape}. Expected shape: {expected_mask_shape}"
    print(f"Mask shape: {mask_data.shape} - OK")