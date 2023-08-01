## Bedrooms
data=data/bedrooms.zip
network_pkl=ckpts/bedrooms.pkl
outdir=out/bedrooms

## Living rooms
# data=data/living_rooms.zip
# network_pkl=ckpts/living_rooms.pkl
# outdir=out/living_rooms

## KITTI
# data=data/kitti
# outdir=out/kitti
# network_pkl=ckpts/kitti.pkl

render_programs=(fake_single)
num_layout_indices=10 # Number of scenes from data/labels/x/boxes.npz
num_coords_seed=20 # Number of camera samples from scene
num_z_seeds=1 # Number of latent codes per scene

# dataset_name=kitti
dataset_name=3dfront
for render_program in ${render_programs[@]}
do
# python generate.py --out_video --outdir=$outdir --data=$data --network_pkl $network_pkl --render_program $render_program --num_layout_indices $num_layout_indices --dataset_name=$dataset_name --num_coords_seed $num_coords_seed --num_z_seeds $num_z_seeds --rand_seed
python generate.py --use_coords_traj --outdir=$outdir --data=$data --network_pkl $network_pkl --render_program $render_program --num_layout_indices $num_layout_indices --dataset_name=$dataset_name --num_coords_seed $num_coords_seed --num_z_seeds $num_z_seeds --rand_seed
done