## A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI
## Getting Started ##
### Installation ###
Our repository is based on the repository [Maniskill2-learn](https://github.com/haosulab/ManiSkill2-Learn)
To get started, enter the parent directory of where you installed [ManiSkill2](https://github.com/haosulab/ManiSkill2) and clone this repo. Assuming the anaconda environment you installed ManiSkill2 is `mani_skill2`, execute the following commands (**note that the ordering is strict**):

```
cd {parent_directory_of_ManiSkill2}
conda activate mani_skill2 #(activate the anaconda env where ManiSkill2 is installed)
git clone https://github.com/haosulab/ManiSkill2-Learn
cd ManiSkill2-Learn
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install pytorch3d
pip install ninja
pip install -e .
pip install protobuf==3.19.0

ln -s ../ManiSkill2/data data # link the ManiSkill2 asset directory to ManiSkill2-Learn
# Alternatively, add `export MS2_ASSET_DIR={path_to_maniskill2}/data` to your bashrc file, so that the OS can find the asset directory no matter where you run MS2 envs.
```

If you would like to use SparseConvNet to perform 3D manipulation learning, install `torchsparse` and its releated dependencies (the `torchsparse` below is forked from the original repo with bug fix and additional normalization functionalities):

```
sudo apt-get install libsparsehash-dev # brew install google-sparsehash if you use Mac OS
pip install torchsparse@git+https://github.com/lz1oceani/torchsparse.git
```
### Training
#### rigid-body
This script is used for`PPO+PointNet` under 'env=PickCube-v0'
```
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py
--work-dir [YOUR_DIR] --gpu-ids 0 --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=pointcloud"
"env_cfg.n_points=1150" "rollout_cfg.num_procs=10" "eval_cfg.num_procs=10"  "env_cfg.reward_mode=dense"
"env_cfg.control_mode=pd_ee_delta_pose" "env_cfg.obs_frame=ee" "eval_cfg.num=100" "eval_cfg.save_traj=False"
"eval_cfg.save_video=True" "train_cfg.n_eval=20000" "train_cfg.n_checkpoint=20000"  "env_cfg.n_goal_points=50"
```
#### soft-body
Before you running the soft-body,you should convert (render) the demonstrations
First, change the controller:
```
python -m mani_skill2.trajectory.replay_trajectory
--traj-path demos/rigid_body/PegInsertionSide-v0/trajectory.h5
--save-traj --target-control-mode pd_ee_delta_pose
--obs-mode none --num-procs 32
```
Then change the obser_moder
```
# Replace `PATH` with appropriate path and `ENV` with appropriate environment name

python tools/convert_state.py --env-name ENV_NAME --num-procs 1 \
--traj-name PATH/trajectory.none.pd_joint_delta_pos.h5 \
--json-name PATH/trajectory.none.pd_joint_delta_pos.json \
--output-name PATH/trajectory.none.pd_joint_delta_pos_pcd.h5 \
--control-mode pd_joint_delta_pos --max-num-traj -1 --obs-mode pointcloud \
--n-points 1200 --obs-frame base --reward-mode dense --render
```
Finally, run the script to train model on 'env=Pour-v0'
```
python maniskill2_learn/apis/run_rl.py configs/brl/bc/pointnet_soft_body.py --work-dir {YOUR_DIR} --gpu-ids 0 --cfg-options
"env_cfg.env_name=Pour-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" "eval_cfg.num=100" "eval_cfg.save_traj=False"
"eval_cfg.save_video=True" "eval_cfg.num_procs=10" "env_cfg.control_mode=pd_ee_delta_pose"
"replay_cfg.buffer_filenames=YOUR_PATH/trajectory.none.pd_ee_delta_pose_pointcloud.h5" "env_cfg.obs_frame=ee"
"train_cfg.n_checkpoint=10000" "replay_cfg.capacity=10000" "replay_cfg.num_samples=-1" "replay_cfg.cache_size=1000" "train_cfg.n_updates=500"
```

Our repository is based on the repository [model](https://drive.google.com/file/d/10_kg4uJSV-PyY4NfDEvopdbC7fOZ9XQU/view?usp=drive_link)




