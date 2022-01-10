# Language-driven Semantic Segmentation (LSeg)
The repo contains official PyTorch Implementation of paper [Language-driven Semantic Segmentation](). 

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

### Running interactive app
Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200.ckpt`. 

Then ``` streamlit run lseg_app.py ```

### Training
Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

``` bash train.sh ```

### Testing
Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

``` bash test.sh ```


### Try demo model
Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200.ckpt`. 

Then follow [lseg_demo.ipynb](https://github.com/isl-org/lang-seg/blob/main/lseg_demo.ipynb) to play around with LSeg. Enjoy!

### Model Zoo
<table>
  <thead>
    <tr style="text-align: right;">
       <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>text encoder</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
       <th>0</th>
       <td>Model for demo</td>
      <th>ViT-L/16</th>
      <th>CLIP ViT-B/32</th>
      <td><a href="https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing">download</a></td>
    </tr>
  </tbody>
</table>

If you find this repo useful, please cite:
```
@article{li2022lan,
  title={Language-driven Semantic Segmentation},
  author={Li, Boyi and Weinberger, Kilian Q and Belongie, Serge and Koltun, Vladlen and Ranftl, Rene},
  journal={arXiv preprint},
  year={2022}
}
```

## Acknowledgement
Thanks to the code base from [DPT](https://github.com/isl-org/DPT), [Pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), [CLIP](https://github.com/openai/CLIP), [Pytorch Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [Streamlit](https://streamlit.io/), [Wandb](https://wandb.ai/site)
