from pickle import TRUE
import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
import torch.nn.functional as F

import math


from skimage.transform import resize
from tqdm import tqdm




def generate_masks(N, s, p1, input_size, savepath):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype("float32")

    masks = np.empty((N, *input_size))

    for i in tqdm(range(N), desc="Generating filters"):
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(
            grid[i], up_size, order=1, mode="reflect", anti_aliasing=False
        )[x: x + input_size[0], y: y + input_size[1]]
    masks = masks.reshape(-1, 1, *input_size)
    masks = torch.from_numpy(masks).float()
    torch.save(masks, savepath)


def MapNormalize(mask):
    return (mask - mask.min()) / (mask.max() - mask.min())


class RISE:
    def __init__(
        self, model: callable, maskpath, device="cpu", one_image=False, **kwargs
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.one_image = one_image
        self.path = maskpath

    def forward(
        self,
        input: torch.Tensor,
        num_basis,
        class_idx=None,
    ):
        self.input_ = input
        # Set chosen class index. By default, the predicted class.
        if class_idx is None:
            with torch.no_grad():
                logits = self.model(self.input_)
                self.class_idx = logits.max(1)[-1]
        else:
            self.class_idx = class_idx
        significant_map, score = self.MapWithoutSpecifiedGranularity(num_basis)
        return (significant_map, score)

    def MapWithoutSpecifiedGranularity(self, num_basis):
        BasisPool = torch.load(self.path)
        map_, score = self.GenerateOneMap(num_basis, BasisPool)
        return map_, score

    def GenerateOneMap(self, num_basis, BasisPool):
        mask_idx = torch.randperm(len(BasisPool))[:num_basis]
        mask_list = BasisPool[mask_idx]
        score_list = self.MaskEvaluation(mask_list)
        # weighted average of all basis, reshape to 4-dimension.
        Map = torch.sum(
            score_list.reshape(-1, 1, 1, 1) * mask_list.cpu(), dim=0, keepdim=True
        )
        Map = MapNormalize(Map)
        score = self.MaskEvaluation(Map).item()
        return Map, score

    def MaskEvaluation(self, mask_list):
        batch_size = 32
        N = mask_list.shape[0]
        score_list = torch.zeros(N)
        # divide batch
        last_idx = -1
        for i in range(math.floor(N / batch_size)):
            mask_batch = mask_list[i * batch_size: (i + 1) * batch_size]
            target_score = self.BatchScoreComputation(mask_batch)
            score_list[i * batch_size: (i + 1) * batch_size] = target_score
            last_idx = i
        # last batch
        if (last_idx + 1) * batch_size < N:
            mask_batch = mask_list[(last_idx + 1) * batch_size:]
            target_score = self.BatchScoreComputation(mask_batch)
            score_list[(last_idx + 1) * batch_size:] = target_score
        return score_list

    def BatchScoreComputation(self, mask_batch: torch.Tensor):
        with torch.no_grad():
            mask_batch = mask_batch.to(self.device)
            logits = self.model(self.input_ * mask_batch)
            score_vector = F.softmax(logits, dim=1)
            target_score = score_vector[:, self.class_idx].squeeze(1)
        return target_score.cpu()

    def __call__(self, input, num_basis=1000, class_idx=None):
        return self.forward(input, num_basis, class_idx)


