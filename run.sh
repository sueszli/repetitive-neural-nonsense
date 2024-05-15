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

# -------------------------------------------------------------------------- clone submodules

git submodule update --init --recursive
git submodule foreach git pull origin master

# -------------------------------------------------------------------------- install dependencies

if ! command -v python3 &> /dev/null; then echo "python3 missing"; exit 1; fi
if ! command -v pip &> /dev/null; then echo "pip missing"; exit 1; fi

python3 -m pip install --upgrade pip

# dev: find out dependencies
# rm -rf requirements.txt
# pip install pipreqs
# pipreqs .

# install dependencies
pip install black
pip install recommenders
pip install -r requirements.txt
