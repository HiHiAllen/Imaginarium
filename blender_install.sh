#downliad blender
# wget https://mirror.freedif.org/blender/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz
#unzip
# tar -xvf ./third_party/blender-4.3.2-linux-x64.tar.xz


cd  ./third_party/blender-4.3.2-linux-x64

#inset to path
BLENDER_PATH=$(pwd)
echo "export PATH=$BLENDER_PATH:$PATH" >> ~/.bashrc

#instal python requirment
cd ./4.3/python/bin
#enable pip
./python3.11 -m ensurepip

#install dep
./python3.11 -m pip install pandas==2.2.2 shapely openpyxl trimesh pyassimp matplotlib scipy wheel hydra-core tqdm opencv-python openai pyyaml cython networkx pillow
./python3.11 -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
./python3.11 -m pip install "numpy<2.0" --force-reinstall

apt-get install libassimp-dev # yum install libassimp-devel
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
