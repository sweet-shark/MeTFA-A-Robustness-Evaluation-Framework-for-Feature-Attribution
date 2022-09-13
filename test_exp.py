import torch
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
import os
import argparse
import numpy as np
import random

from MeTFA_core import base_MeTFA
from Gradient import GradientMask
from util import load_image, basic_visualize, denormalize, normalize, load_model_new,preprocess_image
from RISE import RISE, generate_masks
from cam.scorecam import ScoreCAM
from IGOS_generate_video import Get_blurred_img, Integrated_Mask
from lime import lime_image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class RISE_MeTFA(base_MeTFA):
    def __init__(self, **args):
        super().__init__(**args)
        self.explanation = 'RISE'

    def GetBaseExplanation(self, input_image, num_basis=1000, class_idx=None):
        self.model.eval()
        maskspath = "RISEMask.pt"
        if input_image.device != self.device:
            input_image = input_image.to(self.device)
        if not os.path.isfile(maskspath):
            generate_masks(10000, 8, 0.1, (224, 224), "RISEMask.pt")
        rise = RISE(self.model, maskspath, self.device)
        map_, score = rise(input_image, num_basis=num_basis,
                           class_idx=class_idx)
        return map_.squeeze(0)


class Grad_MeTFA(base_MeTFA):
    def __init__(self, **args):
        super().__init__(**args)
        self.explanation = 'Grad'

    def GetBaseExplanation(self, input_image, class_idx=None):
        if input_image.device != self.device:
            input_image = input_image.to(self.device)
        return GradientMask(self.model, input_image, class_idx=class_idx)


class ScoreCAM_MeTFA(base_MeTFA):
    def __init__(self, **args):
        super().__init__(**args)
        self.explanation = 'ScoreCAM'

    def GetBaseExplanation(self, input_image, class_idx=None):
        with torch.no_grad():
            self.model.eval()
            model_dict = dict(
                type="vgg16", arch=self.model, layer_name="features_29", input_size=(224, 224)
            )
            scorecam = ScoreCAM(model_dict)
            scorecam_map, cam_final_logit, cam_final_score = scorecam(
                input_image.cuda(), class_idx=class_idx)
            return scorecam_map.squeeze(0)


class IGOS_MeTFA(base_MeTFA):
    def __init__(self, **args):
        super().__init__(**args)
        self.explanation = 'IGOS'

    def GetBaseExplanation(self, input_image, class_idx=None):
        # class_idx = 981
        img = denormalize(input_image.cpu())[0].permute(1, 2, 0).numpy()
        use_cuda = 1
        if class_idx == None:
            img_label = -1
        else:
            img_label = class_idx
        model = load_model_new(use_cuda=use_cuda, model_name='vgg16')
        img, blurred_img, logitori = Get_blurred_img(img, img_label, model, resize_shape=(224, 224),
                                                     Gaussian_param=[51, 50],
                                                     Median_param=11, blur_type='Gaussian', use_cuda=use_cuda)

        mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category = Integrated_Mask(img, blurred_img, model,
                                                                                             img_label,
                                                                                             max_iterations=15,
                                                                                             integ_iter=20,
                                                                                             tv_beta=2,
                                                                                             l1_coeff=0.01 * 100,
                                                                                             tv_coeff=0.2 * 100,
                                                                                             size_init=28,
                                                                                             use_cuda=1)
        if upsampled_mask.max()>upsampled_mask.min():
            upsampled_mask = (upsampled_mask-upsampled_mask.min()) / \
                (upsampled_mask.max()-upsampled_mask.min())
        return (1-upsampled_mask).squeeze(0)


class LIME_MeTFA(base_MeTFA):
    def __init__(self, **args):
        super().__init__(**args)
        self.explanation = 'LIME'

    def get_pil_transform():
        transf = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.CenterCrop(224)]
        )

        return transf

    def batch_predict(self, images):
        apply_transforms = transforms.ToTensor()
        batch = torch.stack(tuple(apply_transforms(i) for i in images), dim=0)
        batch = batch.float().to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def GetBaseExplanation(self, input_image, class_idx=None):
        self.model.eval()
        explainer = lime_image.LimeImageExplainer()
        input_image = np.array(
            input_image.cpu().squeeze(0).permute(1, 2, 0).clone().detach(), dtype=np.float64
        )
        if class_idx is None:
            explanation = explainer.explain_instance(
                input_image,
                self.batch_predict,  # classification function
                top_labels=1,
                hide_color=0,
                num_samples=1000,
                progress_bar=False,
                batch_size=200,
            )
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False,
            )
        else:
            explanation = explainer.explain_instance(
                input_image,
                self.batch_predict,  # classification function
                labels=(class_idx,),
                top_labels=None,
                hide_color=0,
                num_samples=1000,
                progress_bar=False,
                batch_size=200,
            )
            temp, mask = explanation.get_image_and_mask(
                class_idx,
                positive_only=True,
                num_features=5,
                hide_rest=False,
            )
        mask = torch.tensor(mask)
        return mask.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='compare stability of base_explanation and MeTFA-smoothed explanation')
    parser.add_argument(
        '--outer_noise',
        choices=['Normal', 'Uniform', 'Darken', 'Laplace', 'None'],
        default='None',
        help='the noise of eviroment')
    parser.add_argument(
        '--inner_noise',
        choices=['Normal', 'Uniform', 'Darken'],
        default='Uniform',
        help='the sampling distribution of MeTFA')
    parser.add_argument(
        '--base_explanation',
        choices=['Grad', 'RISE', 'LIME', 'IGOS', 'ScoreCAM'],
        default='RISE',
        help='the base explanation')
    parser.add_argument(
        '--repeat',
        type=int,
        default=10,
        help='sampling number of MeTFA')
    parser.add_argument(
        '--model',
        choices=['resnet50', 'densenet169', 'vgg16'],
        default='resnet50',
        help='the target model')
    parser.add_argument(
        '--mode',
        choices=['MeTFA', 'ori'],
        default='ori',
        help='the mode of explanation')
    parser.add_argument(
        '--img_num',
        default='0001',
        help='the target image for exaplanation')
    args = parser.parse_args()
    return args


def Get_metfa(base_explanation, model, repeat, alpha, adaptive, Noise):
    if base_explanation == 'Grad':
        metfa = Grad_MeTFA(model=model, repeats=repeat, alpha=alpha,
                     adaptive=adaptive, Noise=Noise)
    if base_explanation == 'RISE':
        metfa = RISE_MeTFA(model=model, repeats=repeat, alpha=alpha,
                     adaptive=adaptive, Noise=Noise)
    if base_explanation == 'ScoreCAM':
        metfa = ScoreCAM_MeTFA(model=model, repeats=repeat, alpha=alpha,
                         adaptive=adaptive, Noise=Noise)
    if base_explanation == 'LIME':
        metfa = LIME_MeTFA(model=model, repeats=repeat, alpha=alpha,
                     adaptive=adaptive, Noise=Noise)
    if base_explanation == 'IGOS':
        metfa = IGOS_MeTFA(model=model, repeats=repeat, alpha=alpha,
                     adaptive=adaptive, Noise=Noise)
    return metfa


def show_one_explanation(args, class_idx=None, if_save=False,save_pt = False):

    """
    class_idx: if None, then the explanation is for the predicted label
    if_save: if True, then the MeTFA-explanation maps are visualized.
    save_pt: if True, then the MeTFA-explanation maps are saved as .pt file
    """

    #load the config

    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True).to("cuda").eval()
    if args.model == 'densenet169':
        model = models.densenet169(pretrained=True).to("cuda").eval()
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True).to("cuda").eval()
    inner_noise = args.inner_noise
    base_explanation = args.base_explanation
    repeat = args.repeat

    #load the target image
    trans = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ]
    )
    image_name = f'./images/ILSVRC2012_val_0000{args.img_num}.JPEG'
    input_image = load_image(image_name)
    input_image = trans(input_image)
    input_ = input_image.unsqueeze(0)

    #run MeTFA
    metfa = Get_metfa(base_explanation, model, repeat,
                            alpha=0.05, adaptive=False, Noise=inner_noise)
    explanation = metfa.GetBaseExplanation(input_).float()
    is_salient, is_irre = metfa.TestSignificance(input_, 0.1)
    lb, ub = metfa.Compute_Confidence_Interval()

    if metfa.explanation == 'LIME':
        MeTFA_explanation = torch.round(
            metfa.map_list[metfa.k1:(metfa.k2+1)].float().mean(dim=0).unsqueeze(0))

    else:
        MeTFA_explanation = metfa.map_list[metfa.k1:(
            metfa.k2+1)].float().mean(dim=0).unsqueeze(0)
        MeTFA_explanation = (MeTFA_explanation-MeTFA_explanation.min()) / \
            (MeTFA_explanation.max()-MeTFA_explanation.min())
        
    if if_save:
        if not os.path.exists('./save_imgs'):
            os.makedirs('./save_imgs')
        basic_visualize(input_.cpu(), explanation.cpu(
        ), './save_imgs/'+args.img_num+'_'+metfa.explanation+'_'+inner_noise+'.png', alpha=0.5)
        basic_visualize(input_.cpu(), MeTFA_explanation.cpu(
        ), './save_imgs/'+args.img_num+'_'+metfa.explanation+'_MeTFA_'+inner_noise+'.png', alpha=0.5)
        basic_visualize(input_.cpu(), ub.float().cpu(
        ), './save_imgs/'+args.img_num+'_'+metfa.explanation+'_hb_'+inner_noise+'.png', alpha=0.5)
        basic_visualize(input_.cpu(), lb.float().cpu(
        ), './save_imgs/'+args.img_num+'_'+metfa.explanation+'_lb_'+inner_noise+'.png', alpha=0.5)
        basic_visualize(input_.cpu(), (is_salient+0.5*is_irre).cpu(
        ), './save_imgs/'+args.img_num+'_'+metfa.explanation+'_sal_'+inner_noise+'.png', alpha=0.5)

    if save_pt:
        if not os.path.exists('./test'):
            os.makedirs('./test')
        torch.save(input_,f'./test/{args.img_num}_{inner_noise}_image.pt')
        torch.save(MeTFA_explanation,f'./test/{args.img_num}_{inner_noise}_{args.explanation}_MeTFA.pt')
        torch.save(explanation,f'./test/{args.img_num}_{inner_noise}_{args.explanation}.pt')

    return explanation, MeTFA_explanation



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    setup_seed(10)
    args = parse_args()
    show_one_explanation(args,if_save=True)

    