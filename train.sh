dataset_name=3dfront
# dataset_name=kitti
outdir=out/$dataset_name
data=data/bedrooms.zip
python train.py --outdir=$outdir --cfg=3dfront_2d_volume --data=$data \
--gpus=4 --batch=32 --batch-gpu=8 --mbstd-group=4 --gamma=10 --blur_fade_kimg=0 \
--aug=diff --feature-resolution=128 --density_reg=0.25 \
--use-semantic-loss=True --use-semantic-floor=True --semantic-resolution=64 \
--n_hidden_layers_mlp=1 --hidden_decoder_mlp_dim=64 \
--super_res_shared=True --conv_head_mod=True --dataset_name=$dataset_name