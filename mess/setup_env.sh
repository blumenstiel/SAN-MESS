# run script with
# bash mess/setup_env.sh

# Create new environment "san"
conda create --name san -y python=3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate san

# install requirements from SAN
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas