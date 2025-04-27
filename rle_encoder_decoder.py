import numpy as np

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    
    Returns:
        str: RLE-encoded string.
    """
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format

def rle_decode(mask_rle, shape):
    """
    Decodes an RLE-encoded string into a binary mask.
    
    Args:
        mask_rle (str): RLE-encoded string.
        shape (tuple): (height, width) of the output mask.
    
    Returns:
        np.ndarray: Decoded binary mask.
    """
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]  # Separate start positions and lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)  # Create a flat mask
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1  # Fill mask with 1s

    return mask.reshape(shape, order='F')  # Reshape in column-major order


def generate_random_mask(shape, probability=0.5):
    """
    Generates a random binary mask.

    Args:
        shape (tuple): (height, width) of the mask.
        probability (float): Probability of a pixel being 1 (default is 0.5).

    Returns:
        np.ndarray: Random binary mask.
    """
    return (np.random.rand(*shape) < probability).astype(np.uint8)


if __name__ == "__main__":
    mask_shape = (9, 2)
    random_mask = generate_random_mask(mask_shape, probability=0.5)

    print("Random mask:")
    print(random_mask)

    rle_string = rle_encode(random_mask)
    print("\nRLE-encoded string:")
    print(rle_string)
    decoded_mask = rle_decode(rle_string, mask_shape)
    assert np.all(decoded_mask == random_mask), "Decoding is not the inverse of encoding!"