
CUDA_VISIBLE_DEVICES=$3 python eval_lpips.py --eval_path $1 --reference_path $2

python -m pytorch_fid $1 $2  --device cuda:$3
