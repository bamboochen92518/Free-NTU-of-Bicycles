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

```
conda activate streetunveiler
python preprocess_video.py --video_path data/20250512_164106.mp4 --output_dir data/pandaset_ours_hq/raw/ours/camera/front_camera/ --interval 0.1
```

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

### Step 8. Prepare for Unveiling

#### Inpainting Pretrained Model Preparation

We provide utility modules to simplify the use of inpainting models:

* `utils/zits_utils.py`
* `utils/leftrefill_utils.py`

Please download the pretrained models and place them in the corresponding directories before use.

**ZITS-PlusPlus**

Navigate to `./3rd_party/ZITS-PlusPlus`. Follow the official instructions to download the pretrained model.

Alternatively, you can download and extract the model from this backup Google Drive link, and place the files under `./3rd_party/ZITS-PlusPlus/ckpts`.

You may also run the following commands to download and extract the model:

```bash
cd 3rd_party/ZITS-PlusPlus
mkdir ckpts
cd ckpts
wget https://huggingface.co/jingwei-xu-00/pretrained_backup_for_streetunveiler/resolve/main/ZITS%2B%2B/best_lsm_hawp.pth
wget https://huggingface.co/jingwei-xu-00/pretrained_backup_for_streetunveiler/blob/main/ZITS%2B%2B/model_512.zip
unzip model_512.zip
```

The directory structure should look like this:

```
ZITS-PlusPlus
└── ckpts
    ├── best_lsm_hawp.pth
    ├── model_512
    │   ├── config.yml
    │   ├── models
    │   │   └── last.ckpt
    │   ├── samples
    │   └── validation
    └── model_512.zip
```

Finally, build the NMS module:

```bash
cd 3rd_party/ZITS-PlusPlus/nms/cxx/src 
source build.sh
```

**LeftRefill**

Navigate to `./3rd_party/LeftRefill`. Follow the official instructions to download the pretrained model.

Alternatively, download it from this backup Google Drive link and place it under `./3rd_party/LeftRefill/pretrained_models`.

You may also use the following commands:

```bash
cd 3rd_party/LeftRefill
mkdir pretrained_models
cd pretrained_models
wget https://huggingface.co/jingwei-xu-00/pretrained_backup_for_streetunveiler/resolve/main/LeftRefill/512-inpainting-ema.ckpt
```

The directory structure should look like this:

```
LeftRefill
└── pretrained_models
    └── 512-inpainting-ema.ckpt
```

Update: We provide a simple API wrapper for LeftRefill at [https://github.com/DavidXu-JJ/simple-leftrefill-inpainting](https://github.com/DavidXu-JJ/simple-leftrefill-inpainting). You may use this if you're integrating LeftRefill into your own project.

After downloading the pretrained models, you can run the following command:

```bash
# Example usage:
# sh unveil_preprocess.sh [model_path] [gpu_id] [iteration]
sh unveil_preprocess.sh ./output_pandaset/027/ 0 50000
```


#### Step 9. Unveiling

```bash
# Example usage:
# sh unveil.sh [model_path] [key_frame_list] [iteration] [gpu_id]
sh unveil.sh ./output_pandaset/027_3x/ "150 120 90 60 30 0" 50000 0
```

#### Step 10. Evaluation

```bash
# Example usage:
# sh eval_lpips_fid.sh [model_path] [gt_path] [gpu_id]
sh eval_lpips_fid.sh "output_pandaset/027_3x/instance_workspace_0/final_renders" "output_pandaset/027_3x/instance_workspace_0/gt" 0
```

### Current Progress

#### Data Structure

```
data/
├── pandaset/             # Without LiDAR
├── pandaset_origin/      # With LiDAR
└── pandaset_ours/        # Without LiDAR (Campus view)
└── pandaset_ours_hq/     # Without LiDAR (Campus view), generate by preprocess_video.py
```

> The base path for all files is: `/tmp2/b10902005/StreetUnveiler`

#### Step 7: Output Results

```
output_pandaset/
├── 027/                  # With LiDAR, 3 views
├── 027_1x/               # Without LiDAR, 1 view
├── 027_3x/               # Without LiDAR, 1 view
├── ours/                 # Without LiDAR, 1 view (intended for high quality)
└── ours_low_quality/     # Same as "ours", current version is low quality
```

> I aim to **enhance the quality** of the output in the `ours/` folder, as it's currently equivalent to `ours_low_quality`.

Note: Each output directory (except 027/) contains a training_metrics.json file, which records PSNR and loss values every 1000 iterations during training. Visualization code is not yet implemented. 

#### Step 10: Evaluation Metrics

* Evaluated on `027_3x` (without LiDAR, 1 view):

  * **FID:** 175.61
  * **Average LPIPS:** 0.3733

> These results deviate significantly from those reported in the original paper, likely due to the absence of LiDAR input.

#### Next Steps

I plan to **rebuild the results using LiDAR data**. Since all code has been modified to work without LiDAR, I will likely create a **new branch** for the LiDAR-based version.

### TODO

- [x] Unveiling (step 8 and 9)
- [x] Enhance resolution
- [x] Add video clipping code
- [x] Upload result of step 7
- [ ] Run test for high resolution video
- [ ] Create a new branch for the LiDAR-based version
- [ ] Run test for the LiDAR-based version
- [ ] Add visualize code to show learning curve

### Reference

1. https://github.com/DavidXu-JJ/StreetUnveiler/tree/3804a06ba5720d76db86b049694024dfa5653e3b
2. https://github.com/facebookresearch/vggt