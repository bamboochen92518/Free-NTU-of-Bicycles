import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import json
from scipy.spatial.transform import Rotation as R
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process images for camera pose estimation')
parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images')
parser.add_argument('--time-interval', type=float, default=1.0, help='Time interval between images in seconds')
args = parser.parse_args()

# Get list of image files from directory
image_dir = args.input_dir
time_interval = args.time_interval
image_extensions = ('.jpg', '.jpeg', '.png')
image_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) 
               and f.lower().endswith(image_extensions)]

# Generate timestamps
image_count = len(image_names)
timestamps = [i * time_interval for i in range(image_count)]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize model
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

intrinsic = []
extrinsic = []
current_image_names = []

for i, image_path in enumerate(image_names):
    current_image_names.append(image_path)
    
    if (i + 1) % 10 == 0 or (i + 1) == len(image_names):
        images = load_and_preprocess_images(current_image_names).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            extri, intri = extri.squeeze(0), intri.squeeze(0)

        for e in extri:
            rotation_matrix = e[:, :3]
            translation = e[:, 3]
            rot = R.from_matrix(rotation_matrix.cpu().numpy())
            quat = rot.as_quat()
            e_result = {
                "position": {
                    "x": float(translation[0]),
                    "y": float(translation[1]),
                    "z": float(translation[2]),
                },
                "heading": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                }
            }
            extrinsic.append(e_result)

        for i in intri:
            i_result = {
                "fx": float(i[0, 0]),
                "fy": float(i[1, 1]),
                "cx": float(i[0, 2]),
                "cy": float(i[1, 2])
            }
            intrinsic.append(i_result)

        current_image_names = []

# Ensure output directory exists
os.makedirs(image_dir, exist_ok=True)

# Save results
with open(os.path.join(image_dir, 'poses.json'), 'w') as f:
    json.dump(extrinsic, f)

with open(os.path.join(image_dir, 'intrinsics.json'), 'w') as f:
    json.dump(intrinsic[0], f)

with open(os.path.join(image_dir, 'timestamps.json'), 'w') as f:
    json.dump(timestamps, f)