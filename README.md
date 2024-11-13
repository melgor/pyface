 &nbsp;
 &nbsp;
  
<div align="center">
  <img src="PyFace.png" width="400"/>
</div>

<div align="center">
	
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>


Train the Face-Recognition model with ease. 

The library implements old and new Face-Recognition architectures, together with their loss functions.

There is a series of associated blog posts with this repository:
1. [History of Face Recognition: Part 1 - DeepFace](https://medium.com/@melgor89/history-of-face-recognition-part-1-deepface-94da32c5355c)

# Dataset
Thanks to the great library [face.evolve](https://github.com/ZhaoJ9014/face.evoLVe/tree/master), all datasets are available online. The only changes I made:
1. Release LFW in image-based format, vs `.bin` in [face.evolve](https://github.com/ZhaoJ9014/face.evoLVe/tree/master)
2. Add a training file for MS and a validation file for UMD.


|Database|Version|\#Identity|\#Image|Download Link|
|:---:|:----:|:-----:|:-----:|:-----:|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_112x112|5,749|13,233|Google Drive|
|[MS-Celeb-1M](https://arxiv.org/pdf/1607.08221.pdf)|Align_112x112|85,742|5,822,653|[Google Drive](https://drive.google.com/file/d/1X202mvYe5tiXFhOx82z4rPiPogXD435i/view?usp=sharing)|
|[Umdfaces](https://arxiv.org/pdf/1611.01484.pdf)|Align_112x112|8,277|367,888|[Google Drive](https://drive.google.com/file/d/13IDdIMqPCd8h1vWOYBkW6T5bjAxwmxm5/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1UzrBMguV5YLh8aawIodKeQ)|

# Architectures

|Architecture|Year| Implementation link|LFW|UMD|weights|
|:---:|:----:|:----:|:-----:|:-----:|:-----:|
|[DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)|2014|[link](https://github.com/melgor/pyface/blob/main/pyface/models/backbones/deepface.py)|96.5%|49.5%||
|[DeepID2+](https://arxiv.org/abs/1412.1265)|2014|[link](https://github.com/melgor/pyface/blob/main/pyface/models/backbones/deepidplus.py)|98.1%|40.19%||
|[CASIA-Net](https://arxiv.org/abs/1411.7923)|2014|[link](https://github.com/melgor/pyface/blob/main/pyface/models/backbones/casianet.py)|99.02%|59.56%||

