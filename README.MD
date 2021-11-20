## Language-driven Semantic Segmentation 

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
``` streamlit run lseg_app.py ```

### Training
``` bash train.sh ```

### Testing
``` bash test.sh ```

### Model Zoo
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model for demo</td>
      <td><a href="https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing">download</a></td>
    </tr>
  </tbody>
</table>

If you find this repo useful, please cite:
```

```

## Acknowledgement
Thanks to the code base from:
1. DPT: https://github.com/isl-org/DPT 
2. Pytorch_lightning: https://github.com/PyTorchLightning/pytorch-lightning 
3. Clip: https://github.com/openai/CLIP 
4. Pytorch Encoding: https://github.com/zhanghang1989/PyTorch-Encoding 
