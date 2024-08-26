import numpy as np


#checking bold data file loading

expected_bold_shape = (65, 77, 56, 180)


def check_bold_data_shape(bold_data, expected_shape):
    """
    Check if the shape of BOLD data matches the expected shape.
    """
    assert bold_data.shape == expected_shape, f"Unexpected BOLD data shape: {bold_data.shape}. Expected shape: {expected_shape}"
    print(f"BOLD data shape: {bold_data.shape} - OK")


def check_mask_shape(mask_data, expected_shape):
    """
    Check if the shape of the mask data matches the expected shape.
    """
    assert mask_data.shape == expected_shape, f"Unexpected mask shape: {mask_data.shape}. Expected shape: {expected_shape}"
    print(f"Mask shape: {mask_data.shape} - OK")


def check_data_reordering(bold_data_reordered, expected_shape):
    """
    Check if the shape of the reordered bold data matches the expected shape.
    """
    assert bold_data_reordered.shape == expected_shape, f"Unexpected mask shape: {bold_data_reordered.shape}. Expected shape: {expected_shape}"
    print(f"Bold data reordered shape: {bold_data_reordered.shape} - OK")


def check_RDM_len(RDM_array, expected_len):
    """
    Check if the length of the 1D RDM array matches the expected length.
    """
    assert len(RDM_array) == expected_len, f"Unexpected length: {len(RDM_array)}. Expected length: {expected_len}"
    print(f"Length of RDM array: {len(RDM_array)} - OK")

