import torch


def GradientMask(Model, image, class_idx = None):
    Model.eval()
    image = image.clone().detach()
    image.requires_grad = True
    out = Model(image)
    if class_idx is None:
        logit, _ = torch.max(out, dim=1)
    else:
        logit = out[0,class_idx]
    logit.backward()
    saliency, _ = torch.max(torch.abs(image.grad).squeeze(0), dim=0)
    if saliency.max()<1e-6:
        print(saliency.max())
        raise
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency.unsqueeze(0)
