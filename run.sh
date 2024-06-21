# -------------------------------------------------------------------------- decompress data
# validate ./data/* files
if [ ! -d data ]; then echo "data/ directory not found"; exit 1; fi
if ! ls data/*-chunk-* &> /dev/null && ! ls data/*.md5 &> /dev/null; then echo "invalid files found in data/"; exit 1; fi

# create data-merged directory
rm -rf data-merged
mkdir data-merged
echo "created data-merged directory"

# merge chunks into data-merged directory
cat data/*-chunk-* > data-merged/merged.tar.gz
echo "merged chunks into data-merged/merged.tar.gz"

# validate checksum
expected_checksum=$(cat data/*.md5)
actual_checksum=$(md5sum data-merged/merged.tar.gz | awk '{ print $1 }')
if [ $expected_checksum != $actual_checksum ]; then echo "checksum mismatch"; exit 1; fi
echo "checksum matched: $expected_checksum == $actual_checksum"

# untar in data-merged
tar -xzf data-merged/merged.tar.gz -C data-merged
rm data-merged/merged.tar.gz
echo "untarred data-merged/merged.tar.gz"

# -------------------------------------------------------------------------- install conda

brew install --cask miniconda
conda update conda

conda init zsh
conda init bash
exit # restart shell

# disable auto-activation of base environment
conda config --set auto_activate_base false

# emulate different platforms in case python wheels are not available
# conda config --env --set subdir osx-64
# conda config --env --set subdir osx-arm64

# ----------------------------------------------------------------------------- start
conda activate base

conda create --yes --name recsys python=3.11 anaconda
conda activate recsys

pip install transformers==4.37.2
pip install tensorflow==2.15.1
pip install torch==2.0.0
pip install scikit-learn==1.4.0
pip install numpy==2.0.0
pip install polars==0.20.31
pip install pyyaml==6.0.1
pip install tqdm
pip install ebrec==0.0.1
pip install black
pip install recommenders

# ----------------------------------------------------------------------------- stop
conda deactivate
conda remove --yes --name recsys --all
conda env list
