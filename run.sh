source /home/zhousifan/anaconda3/bin/activate
# # conda activate /home/zhousifan/anaconda3/envs/centerpoint
conda activate /home/zhousifan/anaconda3/envs/Openmmlab
export http_proxy=127.0.0.1:8888
export https_proxy=127.0.0.1:8888
export all_proxy=socks5://127.0.0.1:8889
#export https_proxy=localhost:7890
export PYTHONPATH="${PYTHONPATH}:/home/zhousifan/OpenPCDet/"
#ssh key '/home/zhousifan/.ssh'
# export PATH=/usr/local/cuda-11.0/bin:$PATH
# export CUDA_PATH=/usr/local/cuda-11.0
# export CUDA_HOME=/usr/local/cuda-11.0
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
cd tools
# python test.py --cfg_file /home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt /home/zhousifan/OpenPCDet/output/home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --rdzv_endpoint=localhost:18040 train.py --launcher pytorch --cfg_file /home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --fix_random_seed
# sh scripts/dist_train.sh 2 --cfg_file /home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
python train.py --cfg_file /home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/pyramid_pointpillar.yaml
#tensorboard --logdir=/home/zhousifan/OpenPCDet/output/home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/pointpillar/default/tensorboard --port 6120