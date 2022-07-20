CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
--Generator=PixelFolder --img2dis --batch=8 \
--out_path=./outputs/PixelFolder_ffhq_256 \
--path_fid=./outputs/PixelFolder_ffhq_256/FID/ \
$IMAGEPATH
