import numpy as np
from PIL import Image
import argparse

def mask_image_by_label(image_path, mask_path, label_index):
    """
    Load an image and an NPZ mask, set pixels to black (0, 0, 0) where the mask equals the specified label.
    
    Args:
        image_path (str): Path to the input image (e.g., JPG)
        mask_path (str): Path to the NPZ mask file
        label_index (int): The label index to match in the mask (e.g., 0 for 'road')
    
    Returns:
        PIL.Image: Modified image with matching mask areas set to black
    """
    # Load the NPZ file and get the mask array
    npz_file = np.load(mask_path)
    mask = npz_file['arr_0']  # Assuming 'arr_0' is the mask array key
    mask_shape = mask.shape  # Expected: (height, width), e.g., (1063, 1890)
    
    # Load the image
    with Image.open(image_path) as img:
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Get image dimensions (width, height), e.g., (1890, 1063)
        width, height = img.size
        # Convert image to numpy array (RGB)
        img_array = np.array(img)
    
    # Verify dimensions: mask is (H, W), image is (W, H) in terms of width and height
    if mask_shape != (height, width):
        raise ValueError(f"Mask shape {mask_shape} does not match image dimensions ({width}, {height})")
    
    # Create a 2D boolean mask where mask == label_index
    label_mask_2d = (mask == label_index)
    
    # Debug: Check number of pixels to be masked
    num_true = np.sum(label_mask_2d)
    print(f"Number of pixels where mask == {label_index}: {num_true}")
    
    # Verify shapes
    print(f"Image array shape: {img_array.shape}")
    print(f"2D Label mask shape: {label_mask_2d.shape}")
    
    # Set pixels to black (0, 0, 0) where label_mask_2d is True
    # Use 2D mask to index height and width, apply [0, 0, 0] to all channels
    img_array[label_mask_2d, :] = [0, 0, 0]
    
    # Convert back to PIL Image
    modified_img = Image.fromarray(img_array)
    
    # Close the NPZ file
    npz_file.close()
    
    return modified_img

# Define class labels
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

# Set up argument parser
parser = argparse.ArgumentParser(description="Mask an image by setting pixels to black where the mask matches a specified label.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file (e.g., JPG)")
parser.add_argument('--mask_path', type=str, required=True, help="Path to the NPZ mask file")
parser.add_argument('--label_index', type=int, required=True, choices=range(len(CLASSES)),
                    help=f"Label index to match in the mask (0 to {len(CLASSES)-1}), e.g., 0 for 'road'")
parser.add_argument('--output_path', type=str, default='masked_image.jpg',
                    help="Path to save the output masked image (default: 'masked_image.jpg')")

# Parse arguments
args = parser.parse_args()

# Process the image
try:
    result_img = mask_image_by_label(args.image_path, args.mask_path, args.label_index)
    
    # Save the result
    result_img.save(args.output_path)
    print(f"Modified image saved to: {args.output_path}")
    print(f"Areas where mask == {args.label_index} ('{CLASSES[args.label_index]}') are set to black (0, 0, 0)")
except Exception as e:
    print(f"Error: {e}")