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
### A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI ###
![image](https://github.com/xtli12/GXU-LIPE/assets/86363634/de43d184-1a84-4d49-a341-d74c9f58f8e9)

Fig. 1. The trend line of success rate with PickSingleEGAD task (488 de-notes the highest score checkpoint of Line 1)

During the initial stage of fine-tuning, we achieve the high-est score on the test set. However, as the training process con-tinued, we observe a decline in the success rate (see Line 1 in Fig. 1), indicating potential overfitting of the model to the spe-cific task. To address this issue and further explore the poten-tial capacity of the model, we introduce a two-stage fine-tuning strategy. 
In the second stage of our approach, we resume the train-ing process from the highest score checkpoint obtained in the initial stage. However, in addition to resuming training, we make two important adjustments: reducing the batch size and the number of samples in each step. This reduction encour-ages the model to pay more attention to smaller volumes of information. However, it is important to note that setting a smaller batch size and fewer samples in each step can intro-duce more noise into the training process, as it may extract more irrelevant information. Nonetheless, this adjustment helps mitigate the overfitting issue to some extent. 


