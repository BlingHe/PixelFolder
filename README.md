# PixelFolder
This is an official implementation of ECCV 2022 Paper ["PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation"](https://arxiv.org/abs/2204.00833). The proposed PixelFolder outperforms existing state-of-the-art pixel synthesis methods (e.g. CIPS, INR-GAN), while reducing the number of parameters and computational overhead by more than 50%. 
![image](https://user-images.githubusercontent.com/57147752/180027451-cb5ab87c-f80e-42e0-bc97-5eabf42e1cd4.png)

## Usage
### Requirements
- Install `CUDA==10.1` with `cudnn7` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `Pytorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
```
- Install other packages:
```
pip install -r requirements.txt
```


### Data Preparation
1. Please download the FFHQ or LSUN (Church/Cat/Bedroom) dataset and organize the images in `.jpg or .png` format to `DATASET_PATH`. <br>
    FFHQ: https://github.com/NVlabs/ffhq-dataset)<br>
    LSUN Church/Cat/Bedroom: https://github.com/fyu/lsun
  
2. Create lmdb datasets.
```
python prepare_data.py images --out LMDB_PATH --size SIZE DATASET_PATH
```
where `LMDB_PATH` is the path of the output lmdb dataset files, `SIZE` is the target resolution and `DATASET_PATH` is the source image files. 

## Citation
If PixelFolder is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper:
```
@article{he2022pixelfolder,
  title={PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation},
  author={He, Jing and Zhou, Yiyi and Zhang, Qi and Peng, Jun and Shen, Yunhang and Sun, Xiaoshuai and Chen, Chao and Ji, Rongrong},
  journal={arXiv preprint arXiv:2204.00833},
  year={2022}
}
```

## Acknowledgement
Our code is built upon the [CIPS implementation](https://github.com/saic-mdal/CIPS) and [Nvidia-licensed CUDA kernels](https://github.com/NVlabs/stylegan2) (fused_bias_act_kernel.cu, upfirdn2d_kernel.cu).

## TODO
- [ ] Usage
  - [ ] Training
  - [ ] Evaluation
    - [ ] FID\Prevision\Recall\IS
    - [ ] Params\GMACs\Speed
- [ ] Model Performance
- [ ] Pretrained Checkpoints

