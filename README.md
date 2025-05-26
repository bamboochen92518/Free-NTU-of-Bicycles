### Environment Setup

Set up separate environments for **Street Unveiler**, **SegFormer**, and **VGGT** to ensure compatibility and stability.

#### Street Unveiler Environment

```bash
# Create and activate environment
conda create -n streetunveiler python=3.10
conda activate streetunveiler

# Install PyTorch with CUDA support (recommended: CUDA 12.1)
# Note: Some users have reported issues with CUDA 11.8. It is recommended to use CUDA 12.1.
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Initialize submodules
git submodule update --init --recursive

# Install submodules
pip install submodules/superpose3d
pip install submodules/sh_encoder
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization

# Install tiny-cuda-nn Torch bindings
cd submodules/tiny-cuda-nn/bindings/torch
pip install .

# Install Pandaset devkit
cd ../../../pandaset-devkit/python
pip install .
```

#### SegFormer Environment

```bash
# Create and activate environment
conda create -n segformer python=3.8
conda activate segformer

# Install required packages
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48

# Install SegFormer
cd 3rd_party/SegFormer
pip install -e . --user
```

#### VGGT Environment

```bash
# Create and activate virtual environment
python -m venv VGGTenv
source VGGTenv/bin/activate

# Install dependencies
cd 3rd_party/vggt
pip install -r requirements.txt
```

### Data Preparation & Pipeline for Street Unveiler

Follow these steps to prepare your dataset and perform scene reconstruction using Street Unveiler and SegFormer.

#### Step 1. Record Video and Extract Frames

Record a driving video and extract image frames from it.

#### Step 2. Organize Your Data Directory

Structure your dataset as follows:

```
data/
└── pandaset/
    └── raw/
        ├── ours/
        │   └── camera/
        │       └── front_camera/
        │           └── {i}.jpg
        └── Pandaset/   # (Leave this directory empty — required for recognition)
```

- Place all extracted frames into the `front_camera/` directory.
- Each image should be named numerically, e.g., `0.jpg`, `1.jpg`, ...

#### Step 3. Compute Camera Parameters Using VGGT

Use VGGT to compute both **intrinsic** and **extrinsic** parameters for each frame.

```bash
# Activate the VGGT environment
source ../VGGTenv/bin/activate

# Navigate to the VGGT directory
cd 3rd_party/vggt

# Run the camera parameter extraction script
python ../../camera_param.py \
    --input-dir=../../data/pandaset/raw/ours/camera/front_camera \
    --time-interval=0.1
```

This will generate the following files in the same directory as your input images:

- `intrinsics.json` – camera intrinsic parameters
- `poses.json` – camera extrinsic parameters
- `timestamps.json` – synthetic timestamps for each frame

#### Step 4. Run COLMAP for SfM Point Cloud Generation

```bash
# Activate Street Unveiler environment
conda activate streetunveiler

# Enter preprocessing scripts directory
cd ./preprocess_script

# Convert dataset to COLMAP-compatible format
bash pandaset2colmap.sh ../data/pandaset/raw ../data/pandaset/colmap

# Run COLMAP on all scenes
bash run_colmap.sh ../data/pandaset/colmap 0 1 0
```

#### Step 5. Generate Semantic Masks with SegFormer

```bash
# Activate SegFormer environment
conda activate segformer

# Go to the mask generation script directory
cd ./3rd_party/neuralsim/dataio/autonomous_driving/waymo

# Predict masks for the 'images' directory
python extract_masks_after_colmap.py \
    --data_root ../../../../../data/pandaset/colmap \
    --segformer_path ../../../../SegFormer \
    --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
    --rgb_dirname images \
    --mask_dirname images_masks

# Predict masks for the 'input' directory
python extract_masks_after_colmap.py \
    --data_root ../../../../../data/pandaset/colmap \
    --segformer_path ../../../../SegFormer \
    --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
    --rgb_dirname input \
    --mask_dirname input_masks
```

Note: If the number of images in the `images` folder differs from the `input` folder or if you encounter distorted images, try re-running `run_colmap.sh`.

#### Step 6. Make Dataset Recognizable by the Codebase

```bash
cd ./data/pandaset/raw
touch Pandaset
```

#### Step 7. Run Reconstruction and Rendering

**Train & Reconstruct:**

```bash
# Activate Street Unveiler environment
conda activate streetunveiler

CUDA_VISIBLE_DEVICES=3 python train.py \
    -s ./data/pandaset/raw \
    -c ./data/pandaset/colmap/027 \
    -m ./output_pandaset/027 \
    -r 4
```

**Render Output:**

```bash
python render.py -m ./output_pandaset/027
```

#### Step 8. Prepare for Unveiling

```bash
# Example usage:
# sh unveil_prepare.sh [model_path] [gpu_id]
sh unveil_prepare.sh ./output_waymo/segment-1172406780360799916_1660_000_1680_000_with_camera_labels 0
```

#### Step 9. Unveiling

```bash
# Example usage:
# sh unveil.sh [model_path] [key_frame_list] [gpu_id]
sh unveil.sh ./output_waymo/segment-1172406780360799916_1660_000_1680_000_with_camera_labels "150 120 90 60 30 0" 0
```

### TODO

- [ ] Unveiling (step 8 and 9)
- [ ] Enhance resolution
- [ ] Add video clipping code
- [ ] Upload result of step 7

### Reference

1. https://github.com/DavidXu-JJ/StreetUnveiler/tree/3804a06ba5720d76db86b049694024dfa5653e3b
2. https://github.com/facebookresearch/vggt