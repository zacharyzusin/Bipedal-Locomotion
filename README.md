# todo
# debug reward
# inverse kinematics for foot sole
# reference policy should provide joint positions and not abolute positions
# reward pelvis


sudo apt update
sudo apt install -y \
    libx11-6 libgl1 libgl1-mesa-dev libglew-dev \
    libosmesa6 libosmesa6-dev patchelf


cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n rl-mujoco python=3.10 -y
conda activate rl-mujoco

pip install numpy==1.26.4
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU cluster
pip install mujoco==3.1.2
pip install fastapi uvicorn
pip install dataclasses-json
pip install fastapi uvicorn pillow
pip install matplotlib

mkdir -p ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/3.1.2/mujoco-3.1.2-linux-x86_64.tar.gz
tar -xvf mujoco-3.1.2-linux-x86_64.tar.gz -C ~/.mujoco/

export MUJOCO_DIR=$HOME/.mujoco/mujoco-3.1.2
export MUJOCO_GL=osmesa

echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
cd /workspace/simulate-bidped
pip install -e .

git config --global user.name "Paul Kroeger"
git config --global user.email "paul.kroeger@columbia.edu"