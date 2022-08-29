# HOP-VLN-finetune

This respository is the finetune code of [HOP: History-and-Order Aware Pre-training for Vision-and-Language Navigation](https://arxiv.org/abs/2203.11591). The code is based on [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT). Thanks to [Yicong Hong](https://github.com/YicongHong) for releasing the Recurrent-VLN-BERT code.

## Prerequisites
### Installation
- Install docker
  Please check [here](https://docs.docker.com/engine/install/ubuntu/) to install docker.
- Create container
  To pull the image: 
  ```sh
  docker pull starrychiao/hop-recurrent:v1
  ```
  If your CUDA version is 11.3, you can pull the image:
  ```sh
  docker push starrychiao/vlnbert-2022-3090:tagname
  ```
  To create the container:
  ```sh
  docker run -it --ipc host  --shm-size=1024m --gpus all --name your_name  --volume "your_directory":/root/mount/Matterport3DSimulator starrychiao/hop-recurrent:v1
  ```
  or (if you pull the image for cuda 11.3)
  ```sh
  docker run -it --ipc host  --shm-size=1024m --gpus all --name your_name  --volume "your_directory":/root/mount/Matterport3DSimulator starrychiao/vlnbert-2022-3090:tagname
  ```
- Set up
  ```sh
  docker start "your container id or name"
  docker exec -it "your container id or name" /bin/bash
  cd /root/mount/Matterport3DSimulator
  ```
- Download the trained models.

## R2R
### Data Preparation
Please follow the instructions below to prepare the data in directories:
- MP3D navigability graphs: `connectivity`
    - Download the [connectivity maps ](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- MP3D image features: `img_features`
    - Download the [Scene features](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365).
- R2R data: `data`
    - Download the [R2R data [5.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R/data).
- Augmented data: `data/prevalent`
    - Download the [collected triplets in PREVALENT [1.5GB]](https://zenodo.org/record/4437864/files/prevalent_aug.json?download=1) (pre-processed for easy use).

## NDH

```sh
cd finetune_ndh
```

### Data Preparation
Please follow the instructions below to prepare the data in directories:
- MP3D navigability graphs: `connectivity`
    - Download the [connectivity maps ](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- MP3D image features: `img_features`
    - Download the [Scene features](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365).
    
### Initial HOP weights
- Pre-trained HOP weights: `load/model`
  - Download the `pytorch_model.bin` from [here](https://drive.google.com/drive/folders/1BbFUns4CqIDMfAJ_fPccRrAiOQ8VAsGh?usp=sharing).

### Training
```bash
bash run/train.bash
```
### Evaluating
```bash
bash run/test.bash
```
## Citation
If you use or discuss our HOP, please cite our paper:
```
@InProceedings{Qiao2022HOP,
    author    = {Qiao, Yanyuan, Qi Yuankai, Hong, Yicong, Yu, Zheng, Wang, Peng and Wu, Qi},
    title     = {HOP: History-and-Order Aware Pre-training for Vision-and-Language Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {15418-15427}
}
```
