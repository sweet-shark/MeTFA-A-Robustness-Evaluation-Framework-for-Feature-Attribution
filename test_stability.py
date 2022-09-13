import torch
import tqdm
import os
import argparse
import torchvision.models as models
import torchvision.transforms as transforms

from test_exp import Get_st
from util import load_image
from open_source.MeTFA_core import add_noise

def parse_args():
    parser = argparse.ArgumentParser(
        description='compare stability of base_explanation and MeTFA-smoothed explanation')
    parser.add_argument(
        '--outer_noise',
        choices=['Normal', 'Uniform', 'Darken', 'Laplace', 'None'],
        default='Uniform',
        help='the noise of eviroment')
    parser.add_argument(
        '--inner_noise',
        choices=['Normal', 'Uniform', 'Darken'],
        default='Uniform',
        help='the sampling distribution of MeTFA')
    parser.add_argument(
        '--base_explanation',
        choices=['Grad', 'RISE','ScoreCAM','IGOS','LIME'],
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
        '--alpha',
        default='005',
        help='the alpha')
    args = parser.parse_args()
    return args


def main(outer_noise, inner_noise, base_explanation, repeat, model, alpha, test_image_number = 4):
    #compare store the explanations of MeTFA-smoothed explanation, SmoothGrad-smoothed explanation and the original explanation to compare the stability of them.
    compare = {} 
    if os.path.exists(f'./stability/Compare_{base_explanation}_{outer_noise}_{inner_noise}_{repeat}_{model}_{alpha}.pt'):
        compare = torch.load(
            f'./stability/Compare_{base_explanation}_{outer_noise}_{inner_noise}_{repeat}_{model}_{alpha}.pt')
        print(
            f'compare {inner_noise}, {outer_noise}, {model}, {repeat}, {alpha} has existed.')
    else:
        print(
            f'./stability/Compare_{base_explanation}_{outer_noise}_{inner_noise}_{repeat}_{model}_{alpha}.pt')
        alpha1 = float(alpha[0] + '.' + alpha[1:])
        print('alpha = ', alpha1)
        for mode in ['MeTFA', 'ori', 'SG']:
            stability_list = []
            for i in tqdm.tqdm(range(1, test_image_number+1)):
                mask_list = []
                for stability in range(10):
                    img_num = f"{i:0>4}"
                    if mode == 'MeTFA':
                        save_inf = torch.load(
                            f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{inner_noise}_MeTFA_{repeat}_{model}.pt')
                        map_list = save_inf['map_list'].float()
                        map_flatten = map_list.reshape(map_list.shape[0],-1)
                        map_flatten_max,_ = map_flatten.max(dim=1)
                        map_flatten_min,_ = map_flatten.min(dim=1)
                        normal_map = (map_flatten-map_flatten_min.unsqueeze(1))/(map_flatten_max.unsqueeze(1)-map_flatten_min.unsqueeze(1)+1e-8)
                        sort_map_list,_ = normal_map.sort(dim=0)
                        k1 = save_inf['k1']
                        k2 = save_inf['k2']
                        MeTFA_smoothed = sort_map_list[k1:(
                            k2+1)].mean(dim=0).unsqueeze(0)
                        MeTFA_smoothed_max,_ = MeTFA_smoothed.max(dim=1)
                        MeTFA_smoothed_min,_ = MeTFA_smoothed.min(dim=1)
                        MeTFA_smoothed = (MeTFA_smoothed-MeTFA_smoothed_min.unsqueeze(1))/(MeTFA_smoothed_max.unsqueeze(1)-MeTFA_smoothed_min.unsqueeze(1)+1e-8)
                        if base_explanation == 'LIME':
                            MeTFA_smoothed = torch.round(MeTFA_smoothed)
                        mask_list.append(MeTFA_smoothed)

                    if mode == 'SG':
                        save_inf = torch.load(
                            f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{inner_noise}_MeTFA_{repeat}_{model}.pt')
                        map_list = save_inf['map_list'].float()
                        map_flatten = map_list.reshape(map_list.shape[0],-1)
                        map_flatten_max,_ = map_flatten.max(dim=1)
                        map_flatten_min,_ = map_flatten.min(dim=1)
                        normal_map = (map_flatten-map_flatten_min.unsqueeze(1))/(map_flatten_max.unsqueeze(1)-map_flatten_min.unsqueeze(1)+1e-8)
                        SG_smoothed = normal_map.mean(dim=0).unsqueeze(0)
                        SG_smoothed_max,_ = SG_smoothed.max(dim=1)
                        SG_smoothed_min,_ = SG_smoothed.min(dim=1)
                        SG_smoothed = (SG_smoothed-SG_smoothed_min.unsqueeze(1))/(SG_smoothed_max.unsqueeze(1)-SG_smoothed_min.unsqueeze(1)+1e-8)
                        if base_explanation == 'LIME':
                            SG_smoothed = torch.round(SG_smoothed)
                        mask_list.append(SG_smoothed)

                    if mode == 'ori':
                        ori_mask = torch.load(
                            f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{mode}_{model}.pt').float()
                        map_flatten = ori_mask.reshape(ori_mask.shape[0],-1)
                        map_flatten_max,_ = map_flatten.max(dim=1)
                        map_flatten_min,_ = map_flatten.min(dim=1)
                        ori_mask = (map_flatten-map_flatten_min.unsqueeze(1))/(map_flatten_max.unsqueeze(1)-map_flatten_min.unsqueeze(1)+1e-8)
                        mask_list.append(ori_mask)

                map_list = torch.cat(mask_list, dim=0)
                stability_list.append(map_list.std(dim=0).mean())
            stability_tensor = torch.tensor(stability_list)
            compare[mode] = stability_tensor
        compare['inf'] = 'this is a dictinary for comparasion of stability of {base_explanation} and {base_explanation}_MeTFA'
        torch.save(
            compare, f'./stability/Compare_{base_explanation}_{outer_noise}_{inner_noise}_{repeat}_{model}_{alpha}.pt')
    st_result = compare['MeTFA']
    ori_result = compare['ori']
    sg_result = compare['SG']
    print('*-'*30)
    print((inner_noise, outer_noise), 'repeat:',
          repeat, 'model:', model, 'alpha:', alpha)
    print('MeTFA vs ori:')
    print(' '*10, (st_result.mean(), ori_result.mean()))
    print(' '*10, (st_result / ori_result).mean())
    print('MeTFA vs SG:')
    print(' '*10, (st_result.mean(), sg_result.mean()))
    print(' '*10, (st_result/sg_result).mean())
    print('SG vs ori:')
    print(' '*10, (sg_result < ori_result).sum())
    print(' '*10, (sg_result/ori_result).mean())
    print('MeTFA:',st_result.mean(),'SG:',sg_result.mean())

def test_stability(args,test_image_number=4):
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
    for mode in ['MeTFA', 'ori']:
        for stability in range(10):
            for i in tqdm.tqdm(range(1, test_image_number+1)):
                img_num = f"{i:0>4}"

                if mode == 'ori':
                    if os.path.exists(f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{mode}_{args.model}.pt'):
                        continue
                if mode == 'MeTFA':
                    if os.path.exists(f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{inner_noise}_{mode}_{repeat}_{args.model}.pt'):
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

                if not os.path.exists('./stability'):
                    os.makedirs('./stability')

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
                        save_inf, f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{inner_noise}_{mode}_{repeat}_{args.model}.pt')
                if mode == 'ori':
                    attribution_map = st.GetBaseExplanation(input_)
                    torch.save(
                        attribution_map, f'./stability/{base_explanation}_{img_num}_{stability}_{outer_noise}_{mode}_{args.model}.pt')
                del st


if __name__ == '__main__':
    args = parse_args()
    base_explanation = args.base_explanation
    repeat = args.repeat
    outer_noise = args.outer_noise
    inner_noise = args.inner_noise
    model = args.model
    alpha = args.alpha
    test_stability(args)
    main(outer_noise, inner_noise, base_explanation, repeat, model, alpha)
