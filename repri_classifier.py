"""
The following code copied from the official RePRI repo.
"""

import torch
import torch.nn.functional as F
from typing import List
from collections import defaultdict
from typing import Tuple
import torch.distributed as dist
from visdom_logger import VisdomLogger

def intersectionAndUnionGPU(preds: torch.tensor,
                            target: torch.tensor,
                            num_classes: int,
                            ignore_index=255) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        preds : shape [H, W]
        target : shape [H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [num_class]
        area_union : shape [num_class]
        area_target : shape [num_class]
    """
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape == target.shape
    preds = preds.view(-1)
    target = target.view(-1)
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]

    # Addind .float() becausue histc not working with long() on CPU
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_intersection
    # print(torch.unique(intersection))
    return area_intersection, area_union, area_target

def batch_intersectionAndUnionGPU(logits: torch.Tensor,
                                  target: torch.Tensor,
                                  num_classes: int,
                                  ignore_index=255) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        logits : shape [n_task, shot, num_class, h, w]
        target : shape [n_task, shot, H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [n_task, shot, num_class]
        area_union : shape [n_task, shot, num_class]
        area_target : shape [n_task, shot, num_class]
    """
    n_task, shots, num_classes, h, w = logits.size()
    H, W = target.size()[-2:]

    logits = F.interpolate(logits.view(n_task * shots, num_classes, h, w),
                           size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots, num_classes, H, W)
    preds = logits.argmax(2)  # [n_task, shot, H, W]

    n_tasks, shot, num_classes, H, W = logits.size()
    area_intersection = torch.zeros(n_tasks, shot, num_classes)
    area_union = torch.zeros(n_tasks, shot, num_classes)
    area_target = torch.zeros(n_tasks, shot, num_classes)
    for task in range(n_tasks):
        for shot in range(shots):
            i, u, t = intersectionAndUnionGPU(preds[task][shot], target[task][shot],
                                              num_classes, ignore_index=ignore_index) # i,u, t are of size()
            area_intersection[task, shot, :] = i
            area_union[task, shot, :] = u
            area_target[task, shot, :] = t
    return area_intersection, area_union, area_target


def to_one_hot(mask: torch.tensor, num_classes: int) -> torch.tensor:
    """
    inputs:
        mask : shape [n_task, shot, h, w]
        num_classes : Number of classes

    returns :
        one_hot_mask : shape [n_task, shot, num_class, h, w]
    """
    n_tasks, shot, h, w = mask.size()
    one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w).to('cuda' if torch.cuda.is_available() else 'cpu')
    new_mask = mask.unsqueeze(2).clone()
    new_mask[torch.where(new_mask == 255)] = 0  # Ignore_pixels are anyways filtered out in the losses
    one_hot_mask.scatter_(2, new_mask, 1).long()
    return one_hot_mask

class Classifier(object):
    def __init__(self, args):
        self.num_classes = 2
        self.temperature = args.temperature
        self.adapt_iter = args.adapt_iter
        self.weights = args.weights
        self.lr = args.cls_lr
        self.FB_param_update = args.FB_param_update
        self.visdom_freq = args.cls_visdom_freq
        self.FB_param_type = args.FB_param_type
        self.FB_param_noise = args.FB_param_noise

    def init_prototypes(self, features_s: torch.tensor, features_q: torch.tensor, text_s: torch.tensor,
                        gt_s: torch.tensor, gt_q: torch.tensor, subcls: List[int],
                        callback) -> None:
        """
        inputs:
            features_s : shape [n_task, shot, c, h, w]
            features_q : shape [n_task, 1, c, h, w]
            text_s: shape [n_task, texts = 1, c]
            gt_s : shape [n_task, shot, H, W]
            gt_q : shape [n_task, 1, H, W]

        returns :
            prototypes : shape [n_task, c]
            bias : shape [n_task]
        """

        # DownSample support masks
        n_task, shot, c, h, w = features_s.size()
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
        ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_task, shot, 1, h, w]

        # Computing prototypes
        fg_mask = (ds_gt_s == 1) # [n_tasks, shot, 1, 240, 240]

        # ============= NOTE: Proposed Solution with text embedding only Begin =============
        fg_prototype = text_s[:, -1, :]
        # ============= NOTE: Proposed Solution with text embedding only End ===============

        # ============= NOTE: Proposed Solution with text embedding + few-shot support images Begin =============
        # fg_prototype = text_s[:, -1, :]
        # ============= NOTE: Proposed Solution with text embedding + few-shot support images End ===============


        # ============= NOTE: Alternative Solution 1 Start =============

        # # per-shot masked average pooling or per-shot prototype
        # fg_prototype = (features_s * fg_mask).sum(dim=(3, 4))
        # fg_prototype /= (fg_mask.sum(dim=(3, 4)) + 1e-10) # [n_tasks, shot, c]

        # # average the prototype across all shots including the label embeddings
        # fg_prototype = torch.concat([fg_prototype, text_s[:, -1, :].unsqueeze(1)], dim=1) # [n_tasks, shot + 1, c]
        # fg_prototype = torch.mean(fg_prototype, dim=1)

        # ============= NOTE: Alternative Solution 1 End =============

        # store the initial weight / support prototype before finding the logits
        self.prototype = fg_prototype

        # get the logit prediction at each spatial location in each query image (total of n_tasks)
        logits_q = self.get_logits(features_q)  # [n_tasks, shot, h, w], be precise => [n_tasks, 1, h, w]
        self.bias = logits_q.mean(dim=(1, 2, 3))

        assert self.prototype.size() == (n_task, c), self.prototype.size()
        assert torch.isnan(self.prototype).sum() == 0, self.prototype

        if callback is not None:
            self.update_callback(callback, 0, features_s, features_q, subcls, gt_s, gt_q)

    def get_logits(self, features: torch.tensor) -> torch.tensor:

        """
        Computes the cosine similarity between self.prototype and given features
        inputs:
            features : shape [n_tasks, shot, c, h, w]

        returns :
            logits : shape [n_tasks, shot, h, w]
        """

        # Put prototypes and features in the right shape for multiplication
        features = features.permute((0, 1, 3, 4, 2))  # [n_task, shot, h, w, c]
        prototype = self.prototype.unsqueeze(1).unsqueeze(2)  # [n_tasks, 1, 1, c]

        # Compute cosine similarity
        cossim = features.matmul(prototype.unsqueeze(4)).squeeze(4)  # [n_task, shot, h, w]
        cossim /= ((prototype.unsqueeze(3).norm(dim=4) * \
                    features.norm(dim=4)) + 1e-10)  # [n_tasks, shot, h, w]

        return self.temperature * cossim

    def get_probas(self, logits: torch.tensor) -> torch.tensor:
        """
        inputs:
            logits : shape [n_tasks, shot, h, w]

        returns :
            probas : shape [n_tasks, shot, num_classes, h, w]
        """
        logits_fg = logits - self.bias.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [n_tasks, shot, h, w]
        probas_fg = torch.sigmoid(logits_fg).unsqueeze(2)
        probas_bg = 1 - probas_fg
        probas = torch.cat([probas_bg, probas_fg], dim=2)
        return probas

    def compute_FB_param(self, features_q: torch.tensor, gt_q: torch.tensor) -> torch.tensor:
        """
        inputs:
            features_q : shape [n_tasks, shot, c, h, w]
            gt_q : shape [n_tasks, shot, h, w]

        updates :
             self.FB_param : shape [n_tasks, num_classes]
        """
        # downsample the groundtruth query
        ds_gt_q = F.interpolate(gt_q.float(), size=features_q.size()[-2:], mode='nearest').long()
        valid_pixels = (ds_gt_q != 255).unsqueeze(2)  # [n_tasks, shot = 1, num_classes, h, w]
        assert (valid_pixels.sum(dim=(1, 2, 3, 4)) == 0).sum() == 0, valid_pixels.sum(dim=(1, 2, 3, 4))

        one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [n_tasks, shot, num_classes, h, w]

        oracle_FB_param = (valid_pixels * one_hot_gt_q).sum(dim=(1, 3, 4)) / valid_pixels.sum(dim=(1, 3, 4))

        if self.FB_param_type == 'oracle':
            self.FB_param = oracle_FB_param
            # Used to assess influence of delta perturbation
            if self.FB_param_noise != 0:
                perturbed_FB_param = oracle_FB_param
                perturbed_FB_param[:, 1] += self.FB_param_noise * perturbed_FB_param[:, 1]
                perturbed_FB_param = torch.clamp(perturbed_FB_param, 0, 1)
                perturbed_FB_param[:, 0] = 1.0 - perturbed_FB_param[:, 1]
                self.FB_param = perturbed_FB_param

        else:
            logits_q = self.get_logits(features_q)
            probas = self.get_probas(logits_q).detach()
            self.FB_param = (valid_pixels * probas).sum(dim=(1, 3, 4))
            self.FB_param /= valid_pixels.sum(dim=(1, 3, 4))

        # Compute the relative error
        deltas = self.FB_param[:, 1] / oracle_FB_param[:, 1] - 1
        return deltas

    def get_entropies(self,
                      valid_pixels: torch.tensor,
                      probas: torch.tensor,
                      reduction='sum') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        inputs:
            probas : shape [n_tasks, shot, num_class, h, w]
            valid_pixels: shape [n_tasks, shot, h, w]

        returns:
            d_kl : FB proportion kl [n_tasks,]
            cond_entropy : Entropy of predictions [n_tasks,]
            marginal : Current marginal distribution over labels [n_tasks, num_classes]
        """
        n_tasks, shot, num_classes, h, w = probas.size()
        assert (valid_pixels.sum(dim=(1, 2, 3)) == 0).sum() == 0, \
               (valid_pixels.sum(dim=(1, 2, 3)) == 0).sum()  # Make sure all tasks have a least 1 valid pixel

        cond_entropy = - ((valid_pixels.unsqueeze(2) * (probas * torch.log(probas + 1e-10))).sum(2))
        cond_entropy = cond_entropy.sum(dim=(1, 2, 3))
        cond_entropy /= valid_pixels.sum(dim=(1, 2, 3))

        marginal = (valid_pixels.unsqueeze(2) * probas).sum(dim=(1, 3, 4))
        marginal /= valid_pixels.sum(dim=(1, 2, 3)).unsqueeze(1)

        d_kl = (marginal * torch.log(marginal / (self.FB_param + 1e-10))).sum(1)

        if reduction == 'sum':
            cond_entropy = cond_entropy.sum(0)
            d_kl = d_kl.sum(0)
            assert not torch.isnan(cond_entropy), cond_entropy
            assert not torch.isnan(d_kl), d_kl
        elif reduction == 'mean':
            cond_entropy = cond_entropy.mean(0)
            d_kl = d_kl.mean(0)
        return d_kl, cond_entropy, marginal

    def get_ce(self,
                probas: torch.tensor,
                valid_pixels: torch.tensor,
                one_hot_gt: torch.tensor,
                reduction: str = 'sum') -> torch.tensor:
        """
        inputs:
            probas : shape [n_tasks, shot, c, h, w]
            one_hot_gt: shape [n_tasks, shot, num_classes, h, w]
            valid_pixels : shape [n_tasks, shot, h, w]

        updates :
             ce : Cross-Entropy between one_hot_gt and probas, shape [n_tasks,]
        """
        ce = - ((valid_pixels.unsqueeze(2) * (one_hot_gt * torch.log(probas + 1e-10))).sum(2))  # [n_tasks, shot, h, w]
        ce = ce.sum(dim=(1, 2, 3))  # [n_tasks]
        ce /= valid_pixels.sum(dim=(1, 2, 3))
        if reduction == 'sum':
            ce = ce.sum(0)
        elif reduction == 'mean':
            ce = ce.mean(0)
        return ce

    def RePRI(self,
                features_s: torch.tensor,
                features_q: torch.tensor,
                gt_s: torch.tensor,
                gt_q: torch.tensor,
                subcls: List,
                n_shots: torch.tensor,
                callback: VisdomLogger) -> torch.tensor:
        """
        Performs RePRI inference

        inputs:
            features_s : shape [n_tasks, shot, c, h, w]
            features_q : shape [n_tasks, shot, c, h, w]
            gt_s : shape [n_tasks, shot, h, w]
            gt_q : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task
            n_shots : # of support shots for each task, shape [n_tasks,]

        updates :
            prototypes : torch.Tensor of shape [n_tasks, num_class, c]

        returns :
            deltas : Relative error on FB estimation right after first update, for each task,
                     shape [n_tasks,]
        """
        deltas = torch.zeros_like(n_shots)
        l1, l2, l3 = self.weights
        if l2 == 'auto':
            l2 = 1 / n_shots
        else:
            l2 = l2 * torch.ones_like(n_shots)
        if l3 == 'auto':
            l3 = 1 / n_shots
        else:
            l3 = l3 * torch.ones_like(n_shots)
        
        # this self.prototype include the information of the label emdedding to enrich the support prototype
        self.prototype.requires_grad_()
        self.bias.requires_grad_()
        optimizer = torch.optim.SGD([self.prototype, self.bias], lr=self.lr)

        # downsample the groundth truth query and support
        ds_gt_q = F.interpolate(gt_q.float(), size=features_s.size()[-2:], mode='nearest').long()
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.size()[-2:], mode='nearest').long()

        valid_pixels_q = (ds_gt_q != 255).float()  # [n_tasks, shot, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()  # [n_tasks, shot, h, w]

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [n_tasks, shot, num_classes, h, w]

        # self.adapt_iter is a hyperparameter to optimise    
        for iteration in range(1, self.adapt_iter):

            logits_s = self.get_logits(features_s)  # [n_tasks, shot, num_class, h, w]
            logits_q = self.get_logits(features_q)  # [n_tasks, 1, num_class, h, w]
            proba_q = self.get_probas(logits_q)
            proba_s = self.get_probas(logits_s)

            d_kl, cond_entropy, marginal = self.get_entropies(valid_pixels_q,
                                                                proba_q,
                                                                reduction='none')
            ce = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s, reduction='none')
            loss = l1 * ce + l2 * d_kl + l3 * cond_entropy

            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()

            # Update FB_param
            if (iteration + 1) in self.FB_param_update and ('oracle' not in self.FB_param_type) and (l2.sum().item() != 0):
                deltas = self.compute_FB_param(features_q, gt_q).cpu()
                l2 += 1

            if callback is not None and (iteration + 1) % self.visdom_freq == 0:
                self.update_callback(callback, iteration, features_s, features_q, subcls, gt_s, gt_q)
        return deltas

    def get_mIoU(self,
                 probas: torch.tensor,
                 gt: torch.tensor,
                 subcls: torch.tensor,
                 reduction: str = 'mean') -> torch.tensor:
        """
        Computes the mIoU over the current batch of tasks being processed

        inputs:
            probas : shape [n_tasks, shot, num_class, h, w]
            gt : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task


        returns :
            class_IoU : Classwise IoU (or mean of it), shape
        """
        intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt, self.num_classes)  # [num_tasks, shot, num_class]
        inter_count = defaultdict(int)
        union_count = defaultdict(int)

        for i, classes_ in enumerate(subcls):
            inter_count[0] += intersection[i, 0, 0]
            union_count[0] += union[i, 0, 0]
            for j, class_ in enumerate(classes_):
                inter_count[class_] += intersection[i, 0, j + 1]  # Do not count background
                union_count[class_] += union[i, 0, j + 1]
        class_IoU = torch.tensor([inter_count[subcls] / union_count[subcls] for subcls in inter_count if subcls != 0])
        if reduction == 'mean':
            return class_IoU.mean()
        elif reduction == 'none':
            return class_IoU

    def update_callback(self, callback, iteration: int, features_s: torch.tensor,
                        features_q: torch.tensor, subcls: List[int],
                        gt_s: torch.tensor, gt_q: torch.tensor) -> None:
        """
        Updates the visdom callback in case live visualization of metrics is desired

        inputs:
            iteration: Current inference iteration
            features_s : shape [n_tasks, shot, c, h, w]
            features_q : shape [n_tasks, shot, c, h, w]
            gt_s : shape [n_tasks, shot, h, w]
            gt_q : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task


        returns :
            callback : Visdom logger
        """
        logits_q = self.get_logits(features_q)  # [n_tasks, shot, num_class, h, w]
        logits_s = self.get_logits(features_s)  # [n_tasks, shot, num_class, h, w]
        proba_q = self.get_probas(logits_q).detach()  # [n_tasks, shot, num_class, h, w]
        proba_s = self.get_probas(logits_s).detach()  # [n_tasks, shot, num_class, h, w]

        f_resolution = features_s.size()[-2:]
        ds_gt_q = F.interpolate(gt_q.float(), size=f_resolution, mode='nearest').long()
        ds_gt_s = F.interpolate(gt_s.float(), size=f_resolution, mode='nearest').long()

        valid_pixels_q = (ds_gt_q != 255).float()  # [n_tasks, shot, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()  # [n_tasks, shot, h, w]

        one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [n_tasks, shot, num_classes, h, w]
        oracle_FB_param = (valid_pixels_q.unsqueeze(2) * one_hot_gt_q).sum(dim=(1, 3, 4))
        oracle_FB_param /= (valid_pixels_q.unsqueeze(2)).sum(dim=(1, 3, 4))

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [n_tasks, shot, num_classes, h, w]
        ce_s = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s)
        ce_q = self.get_ce(proba_q, valid_pixels_q, one_hot_gt_q)

        mIoU_q = self.get_mIoU(proba_q, gt_q, subcls)

        callback.scalar('mIoU_q', iteration, mIoU_q, title='mIoU')
        if iteration > 0:
            d_kl, cond_entropy, marginal = self.get_entropies(valid_pixels_q,
                                                              proba_q,
                                                              reduction='mean')
            marginal2oracle = (oracle_FB_param * torch.log(oracle_FB_param / marginal + 1e-10)).sum(1).mean()
            FB_param2oracle = (oracle_FB_param * torch.log(oracle_FB_param / self.FB_param + 1e-10)).sum(1).mean()
            callback.scalars(['Cond', 'marginal2oracle', 'FB_param2oracle'], iteration,
                             [cond_entropy, marginal2oracle, FB_param2oracle], title='Entropy')
        callback.scalars(['ce_s', 'ce_q'], iteration, [ce_s, ce_q], title='CE')