[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-pascal-5i-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i-1?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-pascal-5i-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i-5?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-pascal-5i)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-coco-20i-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-1?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-coco-20i-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-5?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-coco-20i-10)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-10?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-fss-1000-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-fss-1000-1?p=hypercorrelation-squeeze-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hypercorrelation-squeeze-for-few-shot/few-shot-semantic-segmentation-on-fss-1000-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-fss-1000-5?p=hypercorrelation-squeeze-for-few-shot)


## Hypercorrelation Squeeze for Few-Shot Segmentation
This is the implementation of the paper "Hypercorrelation Squeeze for Few-Shot Segmentation" by Juhong Min, Dahyun Kang, and Minsu Cho. Implemented on Python 3.7 and Pytorch 1.5.1.

<p align="middle">
    <img src="data/assets/architecture.png">
</p>

For more information, check out project [[website](http://cvlab.postech.ac.kr/research/HSNet/)] and the paper on [[arXiv](https://arxiv.org/abs/2104.01538)].

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

Conda environment settings:
```bash
conda create -n hsnet python=3.7
conda activate hsnet

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```
## Preparing Few-Shot Segmentation Datasets
Download following datasets:

> #### 1. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data):
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from our [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].

> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from our Google Drive: [[train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing)].
> (and locate both train2014/ and val2014/ under annotations/ directory).

> #### 3. FSS-1000
> Download FSS-1000 images and annotations from our [[Google Drive](https://drive.google.com/file/d/1Fn-cUESMMF1pQy8Xff-vPQvXJdZoUlP3/view?usp=sharing)].

Create a directory '../Datasets_HSN' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSSS dataset
    │   ├── model/              # (dir.) implementation of Hypercorrelation Squeeze Network model 
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training HSNet
    │   └── test.py             # code for testing HSNet
    └── Datasets_HSN/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   └── SegmentationClassAug/
        ├── COCO2014/           
        │   ├── annotations/
        │   │   ├── train2014/  # (dir.) training masks (from Google Drive) 
        │   │   ├── val2014/    # (dir.) validation masks (from Google Drive)
        │   │   └── ..some json files..
        │   ├── train2014/
        │   └── val2014/
        └── FSS-1000/           # (dir.) contains 1000 object classes
            ├── abacus/   
            ├── ...
            └── zucchini/

## Training
> ### 1. PASCAL-5<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50, resnet101} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark pascal
>                 --lr 1e-3
>                 --bsz 20
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 2 days until convergence (trained with four 2080 Ti GPUs).


> ### 2. COCO-20<sup>i</sup>
> ```bash
> python train.py --backbone {resnet50, resnet101} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark coco 
>                 --lr 1e-3
>                 --bsz 40
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 1 week until convergence (trained four Titan RTX GPUs).

> ### 3. FSS-1000
> ```bash
> python train.py --backbone {vgg16, resnet50, resnet101} 
>                 --benchmark fss 
>                 --lr 1e-3
>                 --bsz 20
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 3 days until convergence (trained with four 2080 Ti GPUs).

> ### Babysitting training:
> Use tensorboard to babysit training progress:
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 



## Testing

> ### 1. PASCAL-5<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1z4KgjgOu--k6YuIj3qWrGg264GRcMis2?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50, resnet101} 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```


> ### 2. COCO-20<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1WpwmCQzxTWhJD5aLQhsgJASaoxxqmFUk?usp=sharing)].
> ```bash
> python test.py --backbone {resnet50, resnet101} 
>                --fold {0, 1, 2, 3} 
>                --benchmark coco 
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```

> ### 3. FSS-1000
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1JOaaJknGwsrSEPoLF3x6_lDiy4XfAe99?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50, resnet101} 
>                --benchmark fss 
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```

> ### 4. Evaluation *without support feature masking* on PASCAL-5<sup>i</sup>
> * To reproduce the results in Tab.1 of our main paper, **COMMENT OUT line 51 in hsnet.py**: support_feats = self.mask_feature(support_feats, support_mask.clone())
> 
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/18YWMCePIrza194pZvVMqQBuYqhwBmJwd?usp=sharing)].
> ```bash
> python test.py --backbone resnet101 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```


## Visualization

* To visualize mask predictions, add command line argument **--visualize**:
  (prediction results will be saved under vis/ directory)
```bash 
  python test.py '...other arguments...' --visualize  
```

#### Example qualitative results (1-shot):

<p align="middle">
    <img src="data/assets/qualitative_results.png">
</p>
   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@InProceedings{min2021hypercorrelation,
    title={Hypercorrelation Squeeze for Few-Shot Segmentation},
    author={Juhong Min and Dahyun Kang and Minsu Cho},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2021}
}
````
