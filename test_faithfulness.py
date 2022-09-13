import torch
import torchvision.models as models
import tqdm
import os
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse

from util import load_image, apply_transforms, basic_visualize, denormalize
from MeTFA_core import add_noise
from test_exp import Get_st

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ShowImg(TensorImg, dir):
    TensorImg = TensorImg.cpu()
    TensorImg = denormalize(TensorImg)
    new_img_PIL = transforms.ToPILImage()(TensorImg.squeeze()).convert('RGB')
    new_img_PIL.save(dir+'.png')


def Insertion_AUC(model, img_root, saliency_map, shape=(3, 224, 224), if_image=False, ret_pred=False, Noise='None', ret_score_list=False):
    with torch.no_grad():
        if if_image:
            input_ = img_root
        else:
            input_image = load_image(img_root)
            input_ = apply_transforms(input_image)
        if torch.cuda.is_available():
            input_ = input_.cuda()
        score_vector = F.softmax(model(input_)[0], dim=0)
        base_prob, plabel = torch.max(score_vector, dim=0)
        # print('pla',plabel)
        input_ = add_noise(input_, Noise)
        sscore_vector = F.softmax(model(input_)[0], dim=0)
        noise_score = sscore_vector[plabel]
        saliency_flatten = saliency_map.reshape(-1)
        # Add a small perturbation to avoid a whole bunch of the same value in saliency map
        saliency_flatten = saliency_flatten + \
            torch.randn(saliency_flatten.shape)*1e-8
        keep_map = torch.ones(shape[1]*shape[2]).float().cuda()
        _, indices = torch.sort(saliency_flatten, descending=True)

        # Delete 1% pixels according to their saliency values each time
        input_s = torch.zeros(100, *shape).cuda()
        lenth = shape[1]*shape[2]//100
        for i in range(100):
            choose = indices[i*lenth:(i+1)*lenth]
            # delete those pixels
            keep_map[choose] = 0.0
            keep = keep_map.reshape(1, 1, *shape[1:])

            input_i = input_.squeeze(0) * (1-keep).squeeze(0)
            input_s[i, :, :, :] = input_i

        total_score = F.softmax(model(input_s), dim=1)

        total_score = total_score[:, plabel]/noise_score

        # numerical integration via trapezoid formula
        if ret_score_list:
            return total_score
        if ret_pred:
            return torch.sum(total_score[:-1]+total_score[1:])/198, plabel
        return torch.sum(total_score[:-1]+total_score[1:])/198


def Deletion_AUC(model, img_root, saliency_map, shape=(3, 224, 224), if_image=False, ret_pred=False, Noise='None', ret_score_list=False):
    with torch.no_grad():
        if if_image:
            input_ = img_root
        else:
            input_image = load_image(img_root)
            input_ = apply_transforms(input_image)
        if torch.cuda.is_available():
            input_ = input_.cuda()
        score_vector = F.softmax(model(input_)[0], dim=0)
        base_prob, plabel = torch.max(score_vector, dim=0)
        input_ = add_noise(input_, Noise)
        sscore_vector = F.softmax(model(input_)[0], dim=0)
        noise_score = sscore_vector[plabel]
        saliency_flatten = saliency_map.reshape(-1)
        # Add a small perturbation to avoid a whole bunch of the same value in saliency map
        saliency_flatten = saliency_flatten + \
            torch.randn(saliency_flatten.shape)*1e-8
        keep_map = torch.ones(shape[1]*shape[2]).float().cuda()
        _, indices = torch.sort(saliency_flatten, descending=True)

        # Delete 1% pixels according to their saliency values each time
        input_s = torch.zeros(100, *shape).cuda()
        lenth = shape[1]*shape[2]//100
        for i in range(100):
            choose = indices[i*lenth:(i+1)*lenth]
            # delete those pixels
            keep_map[choose] = 0.0
            keep = keep_map.reshape(1, 1, *shape[1:])
            input_i = input_.squeeze(0) * keep.squeeze(0)
            input_s[i, :, :, :] = input_i
        
        total_score = F.softmax(model(input_s), dim=1)
        total_score = total_score[:, plabel]/noise_score

        # numerical integration via trapezoid formula
        if ret_score_list:
            return total_score
        if ret_pred:
            return torch.sum(total_score[:-1]+total_score[1:])/198, plabel
        return torch.sum(total_score[:-1]+total_score[1:])/198

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
        '--metric',
        choices=['vanilla', 'robust'],
        default='vanilla',
        help='vanilla faithfulness or robust faithfulness')
    args = parser.parse_args()
    return args

def test_faithfulness(args,test_image_number=4):
    '''
    generate the map_list for MeTFA and SmoothGrad 
    '''
    print("Model Loading...")
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True).to("cuda").eval()
    if args.model == 'densenet169':
        model = models.densenet169(pretrained=True).to("cuda").eval()
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True).to("cuda").eval()
    outer_noise = args.outer_noise
    inner_noise = args.inner_noise
    base_explanation = args.base_explanation
    print(f'inner:{inner_noise}; outer:{outer_noise}')
    repeat = args.repeat
    for mode in ['MeTFA','ori']:
        for i in tqdm.tqdm(range(1, test_image_number+1)):
            img_num = f"{i:0>4}"
            if mode == 'ori':
                if os.path.exists(f'./faithfulness/{base_explanation}_{img_num}_{outer_noise}_{mode}_{args.model}.pt'):
                    continue
            if mode == 'MeTFA':
                if os.path.exists(f'./faithfulness/{base_explanation}_{img_num}_{outer_noise}_{inner_noise}_{mode}_{repeat}_{args.model}.pt'):
                    continue

            st = Get_st(base_explanation, model, repeat,
                        alpha=0.05, adaptive=False, Noise=inner_noise)

            input_image = load_image(
                f"./images/ILSVRC2012_val_0000{img_num}.JPEG")
            trans = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225]),
                ]
            )
            input_image = trans(input_image)
            input_ = input_image.unsqueeze(0)
            input_ = add_noise(input_, outer_noise)

            if not os.path.exists('./faithfulness'):
                os.makedirs('./faithfulness')

            if mode == 'MeTFA':
                try:
                    is_salient, is_irre = st.TestSignificance(
                        input_, 0.1)
                    lb, ub = st.Compute_Confidence_Interval()
                except:
                    raise
                save_inf = {}
                save_inf['k1'] = st.k1
                save_inf['k2'] = st.k2
                save_inf['map_list'] = st.map_list
                torch.save(
                    save_inf, f'./faithfulness/{base_explanation}_{img_num}_{outer_noise}_{inner_noise}_{mode}_{repeat}_{args.model}.pt')
            if mode == 'ori':
                attribution_map = st.GetBaseExplanation(input_)
                torch.save(
                    attribution_map, f'./faithfulness/{base_explanation}_{img_num}_{outer_noise}_{mode}_{args.model}.pt')
            del st


if __name__ == '__main__':
    setup_seed(10)
    args = parse_args()

    test_faithfulness(args)

    base_explanation = args.base_explanation
    repeat = args.repeat
    outer_noise = args.outer_noise
    inner_noise = args.inner_noise
    model = args.model
    metric = args.metric

    image_root = './images'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('loading model...')
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True).to("cuda").eval()
    if args.model == 'densenet169':
        model = models.densenet169(pretrained=True).to("cuda").eval()
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True).to("cuda").eval()
    
    if metric == 'vanilla':
        epochs = 1
        Noise = 'None'
    else:
        epochs = 10 # compute the metric 10 times and use the average as the approxiamation for the expectation
        Noise = args.outer_noise

    for mode in ['MeTFA','ori']:
        for epoch in range(epochs):
            test_image_number = 4
            del_list = []
            ins_list = []
            if os.path.exists(f'./faithfulness/del_{epoch}_{base_explanation}_{mode}_{Noise}_norm.pt'):
                if os.path.exists(f'./faithfulness/ins_{epoch}_{base_explanation}_{mode}_{Noise}_norm.pt'):
                    continue
            else:
                print(
                    f'./faithfulness/del_{epoch}_{base_explanation}_{mode}_{Noise}_norm.pt')

            for i in tqdm.tqdm(range(1, test_image_number+1)):

                img_num = f"{i:0>4}"
                image_root = f"./images/ILSVRC2012_val_0000{img_num}.JPEG"
                if mode == 'ori':
                    saliency_map = torch.load(
                        f'./faithfulness/{base_explanation}_{img_num}_None_{mode}_{args.model}.pt')
                    del_score = Deletion_AUC(
                        model, image_root, saliency_map.cpu(), Noise='None')
                    ins_score = Insertion_AUC(
                        model, image_root, saliency_map.cpu(), Noise='None')
                else:
                    save_inf = torch.load(
                        f'./faithfulness/{base_explanation}_{img_num}_None_Uniform_{mode}_10_{args.model}.pt')
                    map_list = save_inf['map_list']
                    sort_map_list, _ = torch.sort(map_list, dim=0)
                    k1 = save_inf['k1']
                    k2 = save_inf['k2']
                    saliency_map = sort_map_list[k1:(k2+1)].mean(dim=0)
                    del_score = Deletion_AUC(
                        model, image_root, saliency_map.cpu(), Noise=Noise)
                    ins_score = Insertion_AUC(
                        model, image_root, saliency_map.cpu(), Noise=Noise)

                del_list.append(del_score)
                ins_list.append(ins_score)

            torch.save(
                del_list, f'./faithfulness/del_{epoch}_{base_explanation}_{mode}_{Noise}_norm.pt')
            torch.save(
                ins_list, f'./faithfulness/ins_{epoch}_{base_explanation}_{mode}_{Noise}_norm.pt')


    # -----** start vanilla faithfulness **----
    if metric == 'vanilla':
        MeTFA_del = torch.tensor(torch.load(
            f'./faithfulness/del_0_{base_explanation}_MeTFA_{None}_norm.pt'))
        ori_del = torch.tensor(torch.load(
            f'./faithfulness/del_0_{base_explanation}_ori_{None}_norm.pt'))
        MeTFA_ins = torch.tensor(torch.load(
            f'./faithfulness/ins_0_{base_explanation}_MeTFA_{None}_norm.pt'))
        ori_ins = torch.tensor(torch.load(
            f'./faithfulness/ins_0_{base_explanation}_ori_{None}_norm.pt'))
        print(base_explanation, ':')
        print('             MeTFA      ori')
        print('  deletion:', round(MeTFA_del.mean().item(), 4),
            round(ori_del.mean().item(), 4))
        print('  insertion:', round(MeTFA_ins.mean().item(), 4),
            round(ori_ins.mean().item(), 4))
        print('  overall:', round(MeTFA_ins.mean().item()-MeTFA_del.mean().item(), 4),
            round(ori_ins.mean().item()-ori_del.mean().item(), 4))
    #-----** start robust faithfulness **----
    else:
        MeTFA_del = []
        ori_del = []
        MeTFA_ins = []
        ori_ins = []
        for epoch in range(10):
            MeTFA_del.append(torch.tensor(torch.load(
                f'./faithfulness/del_{epoch}_{base_explanation}_MeTFA_{Noise}_norm.pt')))
            ori_del.append(torch.tensor(torch.load(
                f'./faithfulness/del_{epoch}_{base_explanation}_ori_{Noise}_norm.pt')))
            MeTFA_ins.append(torch.tensor(torch.load(
                f'./faithfulness/ins_{epoch}_{base_explanation}_MeTFA_{Noise}_norm.pt')))
            ori_ins.append(torch.tensor(torch.load(
                f'./faithfulness/ins_{epoch}_{base_explanation}_ori_{Noise}_norm.pt')))
        MeTFA_del = torch.stack(MeTFA_del, dim=0).mean(dim=0)
        ori_del = torch.stack(ori_del, dim=0).mean(dim=0)
        MeTFA_ins = torch.stack(MeTFA_ins, dim=0).mean(dim=0)
        ori_ins = torch.stack(ori_ins, dim=0).mean(dim=0)
        print(base_explanation, ':')
        print('             MeTFA      ori')
        print('  deletion:', round(MeTFA_del.mean().item(), 4),
            round(ori_del.mean().item(), 4))
        print('  insertion:', round(MeTFA_ins.mean().item(), 4),
            round(ori_ins.mean().item(), 4))
        print('  overall:', round(MeTFA_ins.mean().item()-MeTFA_del.mean().item(), 4),
            round(ori_ins.mean().item()-ori_del.mean().item(), 4))

