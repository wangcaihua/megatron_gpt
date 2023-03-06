The is an example of using megatron to pretrain GPT2

# Prepare Env
```
git clone https://github.com/wangcaihua/megatron_gpt.git
git remote set-url origin https://<githubtoken>@github.com/wangcaihua/megatron_gpt.git
```
> https://github.com/settings/tokens

## 1. Ensure you CUDA env is fine
### from the beginning
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
ln -s ~/miniconda3/lib ~/miniconda3/lib64

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
nvcc --version
conda install -c nvidia cuda-nvprof=11.7
```
- download and install miniconda
- using `conda` install pytorch, this will install cuda-11.7
- install `cuda-nvprof` to ensure nvprof is 11.7


### torch has installed
```bash
python -c "import torch; print(torch.__version__)"
# out: 1.13.1+cu117 or 1.13.1
```
that means your `torch` version is 1.13.1, and your required cuda version is 11.7. However, you real cuda version may not be that one. To get the cuda version in you machine, 
```bash
nvcc -V
```
If you get a version like `cuda compilation tools, release 11.7, V11.7.64`, your version of cuda and torch are match. It not, change you cuda version [url](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux). Here is an example:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
rm cuda_11.7.0_515.43.04_linux.run
```

## 2. Install apex and Ninja
First, since `megatron` depend on JIT, Ninja is required to work through:
```bash
sudo apt-get -y install ninja-build
pip install Ninja
```

Second, install apex, which enable you mixed precision and distributed training in Pytorch: 
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd .. && rm -rf apex
```

## 3. Install Megatron
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 285068c8108e0e8e6538f54fe27c3ee86c5217a2
git apply ../megatron.patch
python setup.py bdist_wheel
cd dist
pip install megatron-*
cd ../.. && rm -rf Megatron-LM
```

# Prepare Data
Get `vocab.json` and `merges.txt` from [huggingface](https://huggingface.co/gpt2/tree/main), and rename to:
- vocab.json -> 'gpt2-vocab.json'
- merges.txt -> 'gpt2-merges.txt'

Download data from [openai](https://github.com/openai/gpt-2-output-dataset):
```bash
python download_dataset.py --data_path=gpt2_data 
```

Convert `json` data to IndexDataset:
```bash
bash preprocess_gpt.sh
```

The content of `preprocess_gpt.sh` is as following:
```python
python preprocess_data.py \
       --input gpt2_data/webtext.train.jsonl \
       --output-prefix gpt2_data/my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod \
       --workers 4 \
       --chunk-size 128
```
Before you run `preprocess_gpt.sh`, make sure you change the input path

# Get Model
```bash
# BERT-345M-uncased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
# BERT-345M-cased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
GPT-345M: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip

mkdir ckpt
mv megatron_lm_345m_v0.0.zip ckpt
cd ckpt && unzip megatron_lm_345m_v0.0.zip && rm -rf megatron_lm_345m_v0.0.zip
```

# Pretain
## Run in single GPU
```bash
bash gpt_pretrain.sh
```

## Run on multi-GPU with pipeline/tensor parallel
```bash
bash run_mpi.sh
```
