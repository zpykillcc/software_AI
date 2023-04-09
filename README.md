# Software_AI





## Installation

### Requirements

- Nvidia device with CUDA, [example for Ubuntu 20.04](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux) (if you have no nvidia device, delete [this line](https://github.com/MVIG-SJTU/AlphaPose/blob/master/setup.py#L211) from setup.py
- Python 3.7+
- Cython
- PyTorch 1.11+, for users who want to use 1.5 < PyTorch < 1.11, please switch to the `pytorch<1.11` branch by: `git checkout "pytorch<1.11"`; for users who want to use PyTorch < 1.5, please switch to the `pytorch<1.5` branch by: `git checkout "pytorch<1.5"`
- torchvision 0.12.0+
- numpy
- python-package setuptools >= 40.0, reported by [this issue](https://github.com/MVIG-SJTU/AlphaPose/issues/838)
- Linux, [Windows user check here](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#Windows)

+ 配置conda环境

```shell
conda create -n myconda python=3.8
conda activate alphapose
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```



+ 安装

```shell
# 1. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
pip install cython
sudo apt-get install libyaml-dev
pip install -r requirements.txt
python3 ./AlphaPose/setup.py build develop --user

# 2. Install PyTorch3D (Optional, only for visualization)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable
```



+ 安装ffmpeg

```
sudo apt-get install ffmpeg
```



### Models

#### Alphapose

1. Download the object detection model manually: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place it into `AlphaPose/detector/yolo/data`.
2. (Optional) If you want to use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) as the detector, you can download the weights [here](https://github.com/Megvii-BaseDetection/YOLOX), and place them into `AlphaPose/detector/yolox/data`. We recommend [yolox-l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) and [yolox-x](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth).
3. Download our pose models. Place them into `AlphaPose/pretrained_models`. All models and details are available in our [Model Zoo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md).
4. For pose tracking, please refer to our [tracking docments](https://github.com/MVIG-SJTU/AlphaPose/blob/master/trackers) for model download



#### FaceNet

在https://github.com/zpykillcc/facenet可以找到mtcnn和facenet模型放入recgnize/model_check_point





### 框架

+ flask接受前端请求，调用Alphapose模块和recgnize模块进行姿势识别的人脸识别
+ Alphapose使用pytorch
+ FaceNet使用tensorflow



