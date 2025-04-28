
import pandas as pd
import numpy as np
import pandas.api.types


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def rle_decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """Decodes an RLE-encoded string into a binary mask with validation checks."""
    
    if not isinstance(mask_rle, str) or not mask_rle.strip() or mask_rle.lower() == 'nan':
        # Return all-zero mask if RLE is empty, invalid, or NaN
        return np.zeros(shape, dtype=np.uint8)
    
    try:
        s = list(map(int, mask_rle.split()))
    except:
        raise ParticipantVisibleError("RLE segmentation must contain only integers")
    
    if len(s) % 2 != 0:
        raise ParticipantVisibleError("RLE segmentation must have even-length (start, length) pairs")
    
    if any(x < 0 for x in s):
        raise ParticipantVisibleError("RLE segmentation must not contain negative values")
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    starts, lengths = s[0::2], s[1::2]
    
    for start, length in zip(starts, lengths):
        if start >= mask.size or start + length > mask.size:
            raise ParticipantVisibleError("RLE indices exceed image size")
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='F')  # Convert to column-major order

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    return (2.0 * intersection + 1e-7) / (np.sum(mask1) + np.sum(mask2) + 1e-7)  # Avoid division by zero

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Computes the Dice score between solution and submission."""
    
    # Check if required columns exist
    required_columns = {row_id_column_name, "segmentation"}
    if not required_columns.issubset(solution.columns) or not required_columns.issubset(submission.columns):
        raise ParticipantVisibleError("Solution and submission must contain 'id' and 'segmentation' columns")
    
    # Ensure the IDs match between solution and submission
    if not solution[row_id_column_name].equals(submission[row_id_column_name]):
        raise ParticipantVisibleError("Submission IDs do not match solution IDs")
    
    # Delete the row ID column as Kaggle aligns solution and submission before passing to score()
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    # Decode RLE masks and compute Dice score
    dice_scores = []
    for solution_seg, submission_seg in zip(solution["segmentation"], submission["segmentation"]):
        solution_mask = rle_decode(solution_seg)
        submission_mask = rle_decode(submission_seg)
        dice_scores.append(dice_coefficient(solution_mask, submission_mask))
    
    return np.mean(dice_scores)

