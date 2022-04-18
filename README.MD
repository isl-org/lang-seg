# Language-driven Semantic Segmentation (LSeg)
The repo contains official PyTorch Implementation of paper [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546). 

ICLR 2022

#### Authors: 
* [Boyi Li](https://sites.google.com/site/boyilics/home)
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)
* [Serge Belongie](https://scholar.google.com/citations?user=ORr4XJYAAAAJ&hl=zh-CN)
* [Vladlen Koltun](http://vladlen.info/)
* [Rene Ranftl](https://scholar.google.at/citations?user=cwKg158AAAAJ&hl=de)


### Overview


We present LSeg, a novel model for language-driven semantic image segmentation. LSeg uses a text encoder to compute embeddings of descriptive input labels (e.g., ''grass'' or 'building'') together with a transformer-based image encoder that computes dense per-pixel embeddings of the input image. The image encoder is trained with a contrastive objective to align pixel embeddings to the text embedding of the corresponding semantic class. The text embeddings provide a flexible label representation in which semantically similar labels map to similar regions in the embedding space (e.g., ''cat'' and ''furry''). This allows LSeg to generalize to previously unseen categories at test time, without retraining or even requiring a single additional training sample. We demonstrate that our approach achieves highly competitive zero-shot performance compared to existing zero- and few-shot semantic segmentation methods, and even matches the accuracy of traditional segmentation algorithms when a fixed label set is provided. 

Please check our [Video Demo (4k)](https://www.youtube.com/watch?v=bmU75rsmv6s) to further showcase the capabilities of LSeg.

## Usage
### Installation
Option 1: 

``` pip install -r requirements.txt ```

Option 2: 
```
conda install ipython
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==1.3.5
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install altair
pip install streamlit
pip install --upgrade protobuf
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
```

### Data Preparation
By default, for training, testing and demo, we use [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

```
python prepare_ade20k.py
unzip ../datasets/ADEChallengeData2016.zip
```

Note: for demo, if you want to use random inputs, you can ignore data loading and comment the code at [link](https://github.com/isl-org/lang-seg/blob/main/modules/lseg_module.py#L55). 


### 🌻 Try demo now

#### Download Demo Model
<table>
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>backbone</th>
      <th>text encoder</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
       <td>Model for demo</td>
      <th>ViT-L/16</th>
      <th>CLIP ViT-B/32</th>
      <td><a href="https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing">download</a></td>
    </tr>
  </tbody>
</table>

#### 👉 Option 1: Running interactive app
Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200.ckpt`. 

Then ``` streamlit run lseg_app.py ```

#### 👉 Option 2: Jupyter Notebook
Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200.ckpt`. 

Then follow [lseg_demo.ipynb](https://github.com/isl-org/lang-seg/blob/main/lseg_demo.ipynb) to play around with LSeg. Enjoy!



### Training and Testing Example
Training: Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

``` bash train.sh ```

Testing: Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

``` bash test.sh ```

### Zero-shot Experiments
#### Data Preparation
Please follow [HSNet](https://github.com/juhongm999/hsnet) and put all dataset in `data/Dataset_HSN`

#### Pascal-5i
``` 
for fold in 0 1 2 3; do
python -u test_lseg_zs.py --backbone clip_resnet101 --module clipseg_DPT_test_v2 --dataset pascal \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold ${fold} --nshot 0 \
--weights checkpoints/pascal_fold${fold}.ckpt 
done
```
#### COCO-20i
``` 
for fold in 0 1 2 3; do
python -u test_lseg_zs.py --backbone clip_resnet101 --module clipseg_DPT_test_v2 --dataset coco \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold ${fold} --nshot 0 \
--weights checkpoints/pascal_fold${fold}.ckpt 
done
```
#### FSS
``` 
python -u test_lseg_zs.py --backbone clip_resnet101 --module clipseg_DPT_test_v2 --dataset fss \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold 0 --nshot 0 \
--weights checkpoints/fss_l16.ckpt 
```

``` 
python -u test_lseg_zs.py --backbone clip_vitl16_384 --module clipseg_DPT_test_v2 --dataset fss \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold 0 --nshot 0 \
--weights checkpoints/fss_rn101.ckpt 
```

#### Model Zoo
<table>
  <thead>
    <tr style="text-align: right;">
       <th>dataset</th>
      <th>fold</th>
      <th>backbone</th>
      <th>text encoder</th>
      <th>performance</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
       <th>pascal</th>
       <td>0</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>52.8</th>
      <td><a href="https://drive.google.com/file/d/1gKXh_VORfFBhGVfwaDD-FzLn3hqiIAoR/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>pascal</th>
       <td>1</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>53.8</th>
      <td><a href="https://drive.google.com/file/d/16GrCGJi5hfjMZ0lzd-SHtLvbRpbxa8Sh/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>pascal</th>
       <td>2</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>44.4</th>
      <td><a href="https://drive.google.com/file/d/1hJXCus0QzhphZ3R96fe8WHVYiYfpkF-G/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>pascal</th>
       <td>3</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>38.5</th>
      <td><a href="https://drive.google.com/file/d/1lGzhGxx9FECPg52LAVW_4GZQIMx25ele/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>coco</th>
       <td>0</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>22.1</th>
      <td><a href="https://drive.google.com/file/d/1Q261gloDZQeqVA6pDqWVVzW8I6k0TTHu/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>coco</th>
       <td>1</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>25.1</th>
      <td><a href="https://drive.google.com/file/d/1q7iqEUmPePqqUq0QDUlhHkajTAydvVYn/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>coco</th>
       <td>2</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>24.9</th>
      <td><a href="https://drive.google.com/file/d/1q7iqEUmPePqqUq0QDUlhHkajTAydvVYn/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>coco</th>
       <td>3</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>21.5</th>
      <td><a href="https://drive.google.com/file/d/1H6d6udhJ92eEMeRDF0E5NFJD2LDI1DBX/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>fss</th>
       <td>-</td>
      <th>ResNet101</th>
      <th>CLIP ViT-B/32</th>
      <th>84.7</th>
      <td><a href="https://drive.google.com/file/d/1tIv239sRibgTF73m09Z3fmDYkTngQhwp/view?usp=sharing">download</a></td>
    </tr>
    <tr>
       <th>fss</th>
       <td>-</td>
      <th>ViT-L/16</th>
      <th>CLIP ViT-B/32</th>
      <th>87.8</th>
      <td><a href="https://drive.google.com/file/d/1-wRkHVQ4UxVYiVPNVXFU4Qwo8qliN-xg/view?usp=sharing">download</a></td>
    </tr>
  </tbody>
</table>

If you find this repo useful, please cite:
```
@inproceedings{
li2022languagedriven,
title={Language-driven Semantic Segmentation},
author={Boyi Li and Kilian Q Weinberger and Serge Belongie and Vladlen Koltun and Rene Ranftl},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=RriDjddCLN}
}
```

## Acknowledgement
Thanks to the code base from [DPT](https://github.com/isl-org/DPT), [Pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), [CLIP](https://github.com/openai/CLIP), [Pytorch Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [Streamlit](https://streamlit.io/), [Wandb](https://wandb.ai/site)
