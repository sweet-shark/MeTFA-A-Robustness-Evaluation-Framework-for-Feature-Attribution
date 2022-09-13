  
import torch
import torch.nn.functional as F
from cam.basecam import *
import math

class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_ = input
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            self.class_idx = predicted_class
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            self.class_idx = predicted_class
            score = logit[:, class_idx].squeeze()
        
        logit = F.softmax(logit, dim=-1)

        if torch.cuda.is_available():
            predicted_class= predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()

        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            norm_saliency_maps = torch.zeros(k,1,224,224)
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                
                if saliency_map.max() == saliency_map.min():
                    continue
                
                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                norm_saliency_maps[i:i+1] = norm_saliency_map
                # how much increase if keeping the highlighted region
                # predication on masked input
            score_list = torch.zeros(k)
            # divide batch
            last_idx = -1
            batch_size = 32
            N = k
            for i in range(math.floor(N/batch_size)):
                mask_batch = norm_saliency_maps[i*batch_size:(i+1)*batch_size]
                target_score = self.BatchScoreComputation(mask_batch)
                score_list[i*batch_size:(i+1)*batch_size] = target_score
                last_idx = i
            # last batch
            if (last_idx+1)*batch_size < N:
                mask_batch = norm_saliency_maps[(last_idx+1)*batch_size:]
                target_score = self.BatchScoreComputation(mask_batch)
                score_list[(last_idx+1)*batch_size:] = target_score


            score_saliency_map = torch.sum(score_list.reshape(-1, 1, 1, 1)*norm_saliency_maps.cpu(),
                        dim=0, keepdim=True)
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
        score_saliency_map = score_saliency_map.to(self.device)

        output = self.model_arch(input * score_saliency_map)
        final_logit = output[0][predicted_class]
        final_score = F.softmax(output[0], dim=-1)[predicted_class]
        
        return score_saliency_map.cpu(), final_logit.cpu(), final_score.cpu()



    def BatchScoreComputation(self, mask_batch: torch.Tensor):
        with torch.no_grad():
            mask_batch = mask_batch.to(self.device)
            logits = self.model_arch(self.input_ * mask_batch)
            target_score = F.softmax(logits, dim=1)[:, self.class_idx]
        return target_score.cpu().squeeze(1)

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
