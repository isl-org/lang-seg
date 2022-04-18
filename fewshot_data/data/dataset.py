r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from fewshot_data.data.pascal import DatasetPASCAL
from fewshot_data.data.coco import DatasetCOCO
from fewshot_data.data.fss import DatasetFSS


class FSSDataset:
    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, imagenet_norm=False):
        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        if imagenet_norm:
            cls.img_mean = [0.485, 0.456, 0.406]
            cls.img_std = [0.229, 0.224, 0.225]
            print('use norm: {}, {}'.format(cls.img_mean, cls.img_std))
        else:
            cls.img_mean = [0.5] * 3
            cls.img_std = [0.5] * 3
            print('use norm: {}, {}'.format(cls.img_mean, cls.img_std))

        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
