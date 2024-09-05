from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms
from typing import Tuple
import numpy as np
import matplotlib
import cv2


def tensor_to_img(x:Tensor, size:Tuple=None) -> Image.Image:
    img = transforms.ToPILImage()(x)
    if size is not None:
        img = img.resize(size)
    return img


def heatmap_over_img(x:Tensor, heat: Tensor) -> Image.Image:
    heat = np.array(heat[0,:,:].detach().cpu())
    heat = (255 * (heat - np.min(heat) + 1e-6) / (np.max(heat) - np.min(heat)) + 1e-6).astype(np.uint8)
    color_mapped_heat = cv2.applyColorMap(heat[:, :, np.newaxis], cv2.COLORMAP_JET)
    color_mapped_heat_rgb = cv2.cvtColor(color_mapped_heat, cv2.COLOR_BGR2RGB)
    heat_map = Image.fromarray(color_mapped_heat_rgb,mode='RGB')
    x_img = tensor_to_img(x)
    overlayed_img = Image.blend(heat_map, x_img, 0.5)
    return overlayed_img


def denormalize(x:Tensor, mean_value:Tuple=(111.89, 111.89, 111.89), std_value:Tuple=(27.62, 27.62, 27.62)) -> Tensor:
    output = torch.zeros_like(x)
    output[0,:,:] = x[0,:,:] * std_value[0] + mean_value[0]
    output[1,:,:] = x[1,:,:] * std_value[1] + mean_value[1]
    output[2,:,:] = x[2,:,:] * std_value[2] + mean_value[2]
    return output