## Bedrooms FID dataset
# data=data/bedrooms
# outdir=out/bedrooms
# network_pkl=ckpts/bedrooms.pkl
# python generate_dataset.py --network_pkl $network_pkl --data=$data --outdir=$outdir --num_layout_indices=5515 --num_z_seeds=1 --num_coords_seed=1 --num_images=50000 --rand_coord --start_coords_idx=0

## Living rooms FID dataset
# data=data/living_rooms
# outdir=out/living_rooms
# network_pkl=ckpts/living_rooms.pkl
# python generate_dataset.py --network_pkl $network_pkl --data=$data --outdir=$outdir --num_layout_indices=2613 --num_z_seeds=1 --num_coords_seed=1 --num_images=50000 --rand_coord --start_coords_idx=0

## KITTI FID dataset
# data=data/kitti
# outdir=out/kitti
# network_pkl=ckpts/kitti.pkl
# dataset_name=kitti
# python generate_dataset.py --dataset_name=kitti --network_pkl $network_pkl --data=$data --outdir=$outdir --num_layout_indices=37691 --num_z_seeds=1 --num_coords_seed=1 --num_images=37691 --rand_coord --start_coords_idx=0

# Evaluate FID
# real_data_path=$dataset/images
# python calc_metrics_for_dataset.py --real_data_path $real_data_path --fake_data_path $outdir \
# --resolution $resolution --metrics fid50k_full,kid50k_full --num_runs 1