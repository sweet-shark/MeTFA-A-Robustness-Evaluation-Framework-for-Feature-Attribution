from scipy.special import comb
import numpy as np
import warnings
import torch
import jenkspy
import torchvision

from abc import ABC
from util import denormalize, normalize

def add_noise(image, Noise, level=0.1, denorm=True):
    '''
    add outer noise to simulate real world
    '''
    assert len(image.shape) == 4, 'image must four channels'
    assert Noise in ['Uniform', 'Normal', 'Darken',
                    'Laplace','None'], 'Noise must in [Uniform,Normal,Darken,Laplace,None]'
    test_img = image.detach().clone()
    if denorm:
        test_img = denormalize(test_img)
    if Noise == 'Uniform':
        noise = (torch.rand_like(image, device=image.device) - 0.5)*level*2
        test_img += noise
    if Noise == 'Normal':
        noise = torch.randn(image.shape, device=image.device)*level
        test_img += noise
    if Noise == 'Laplace':
        laplace = torch.distributions.laplace.Laplace(0, level)
        noise = laplace.sample(image.shape).to(image.device)
        test_img += noise
    if Noise == 'Darken':
        ColorJitter = torchvision.transforms.ColorJitter(brightness=level)
        test_img = ColorJitter(test_img)
    test_img = torch.clamp(test_img, min=0, max=1)
    if denorm:
        test_img = normalize(test_img)
    return test_img

class base_MeTFA(ABC):
    """
    A parent class for performing Hard-margin Significance Test
    """

    def __init__(
        self,
        model,
        repeats,
        alpha,
        device="cuda",
        salient_include_thre=0.2,
        irrelevant_include_thre=0.2,
        adaptive=True,
        Noise = 'Uniform'
    ):
        self.model = model.to(device)
        self.repeats = repeats
        self.alpha = alpha
        self.device = device
        self.salient_include_thre = salient_include_thre
        self.irrelevant_include_thre = irrelevant_include_thre
        self.map_list = None
        self.adaptive = adaptive
        assert Noise in ['Uniform','Normal','Darken'], 'inner noise must be uniform or normal'
        self.noise = Noise

    # @abstractmethod
    def GetBaseExplanation(self, input_image):
        """
        Return shape: (1, width, height)
        """
        raise NotImplementedError

    def _TestSalient(self, map_list, repeats, include_thre):
        """
        Get significantly salient regions. Return shape: (1, width, height)
        """
        m = np.arange(repeats)
        N = np.array([repeats for i in range(repeats)])
        prob = np.minimum(0.5, m / N)
        p = comb(N, m) * np.power(prob, m) * np.power(1 - prob, N - m)
        p = np.cumsum(p[::-1])[::-1]
        alpha = self.alpha
        if alpha < np.min(p):
            raise ValueError(
                "Alpha is too small for the specified repeating times. Use a larger alpha or a larger repeats."
            )
        # k is the smallest count for a pixel to be identified significant
        k = np.min(np.where(p < alpha))
        # Get counts for every pixel.
        included = map_list >= include_thre
        count = torch.sum(included, dim=0, keepdim=True)
        is_salient = count >= k
        if torch.sum(is_salient) < 1:
            warnings.warn("No pixels are identified significantly salient.")
        return is_salient

    def _TestIrrelevant(self, map_list, repeats, include_thre):
        """
        Get significantly irrelevant regions. Return shape: (1, width, height)
        """
        m = np.arange(repeats)
        N = np.array([repeats for i in range(repeats)])
        prob = np.maximum(0.5, m / N)
        p = comb(N, m) * np.power(prob, m) * np.power(1 - prob, N - m)
        p = np.cumsum(p)
        alpha = self.alpha
        if alpha < np.min(p):
            raise ValueError(
                "Alpha is too small for the specified repeating times. Use a larger alpha or a larger repeats."
            )
        # k is the largest count for a pixel to be identified significant
        k = np.max(np.where(p < alpha))
        # Get counts for every pixel.
        included = map_list >= include_thre
        count = torch.sum(included, dim=0, keepdim=True)
        is_Irrelevant = count <= k
        if torch.sum(is_Irrelevant) < 1:
            warnings.warn("No pixels are identified significantly Irrelevant.")
        return is_Irrelevant



    def _Generate_Saliency_Maps(self, inputs, perturb_level):
        # obtain original prediction
        self.model.eval()
        inputs = inputs.to(self.device)
        out = self.model(inputs)
        # original_predict = torch.argmax(out, dim=-1)
        original_predict = out
        class_idx = out.max(1)[-1]
        map_list = []
        count, _ = 0, 0
        while count < self.repeats:
            input_ = add_noise(inputs,self.noise)
            count += 1
            explanation = self.GetBaseExplanation(input_, class_idx=class_idx)
            map_list.append(explanation)
        map_list = torch.cat(map_list, dim=0)
        return map_list

    def _JenksCut(self, values):
        cuts = jenkspy.jenks_breaks(
            values[torch.randint(low=0, high=values.shape[-1], size=(1000,))],
            nb_class=2,
        )
        """
        cuts is a three number list [x0,x1,x2].
        x0 is the min of values.
        x1 is the max of the smaller group of the min of the bigger group depending on the codes of jenks.
        x2 is the max of values.
        so sometimes, two of x0, x1, x2 may be equal. for this situation, we set the cut as follows.
        """
        if cuts[0] == cuts[1]:
            cut = cuts[1] + 1e-6
        elif cuts[1] == cuts[2]:
            cut = cuts[1] - 1e-6
        else:
            cut = cuts[1]
        return cut

    def _UpdateThreshold(self, map_list):
        if self.adaptive:
            values = map_list.view(-1)
            cut = self._JenksCut(values)
            self.salient_include_thre = self.irrelevant_include_thre = cut


    def TestSignificance(self, inputs, perturb_level, map_list=None):
        """
        MeTFA with adaptive cut threshold
        """
        if map_list is None:
            map_list = self._Generate_Saliency_Maps(inputs, perturb_level).float()
        map_list += torch.randn(map_list.shape,device = map_list.device)*1e-8
        self._UpdateThreshold(map_list)
        is_salient = self._TestSalient(
            map_list, self.repeats, self.irrelevant_include_thre
        )
        is_irrelevant = self._TestIrrelevant(
            map_list, self.repeats, self.salient_include_thre
        )
        self.is_salient = is_salient
        self.is_irrelevant = is_irrelevant
        self.map_list = map_list
        return is_salient, is_irrelevant


    def Compute_Confidence_Interval(self):
        """
        generate the upper bound and the lower bound map
        """
        map_list = self.map_list
        m = np.arange(self.repeats)
        N = np.array([self.repeats for i in range(self.repeats)])
        prob = 0.5
        p = comb(N, m) * np.power(prob, m) * np.power(1 - prob, N - m)
        p = np.cumsum(p)
        alpha = self.alpha / 2
        if alpha < np.min(p):
            raise ValueError(
                "Alpha is too small for the specified repeating times. Use a larger alpha or a larger repeats."
            )
        k1 = np.max(np.where(p < alpha))
        k2 = self.repeats - k1 - 1
        self.k1 = k1
        self.k2 = k2
        sort_map_list, _ = torch.sort(map_list, dim=0)
        self.sort_map_list = sort_map_list
        lb = sort_map_list[k1:k1+1] 
        ub = sort_map_list[k2:k2+1]  

        return lb, ub
