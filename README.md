##A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI
## Getting Started ##
### Installation ###
Our repository is based on the repository [https://github.com/haosulab/ManiSkill2-Learn]
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
