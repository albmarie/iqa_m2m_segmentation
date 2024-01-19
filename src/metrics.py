import torch
import piq
import piqa
from class_utils import Metric_mode

####################################################################################################
##########          FUNCTIONS          #############################################################
####################################################################################################

def get_metric_fn(metric: Metric_mode, device):
    if metric == Metric_mode.PSNR:
        return lambda tensor_x, tensor_y:     piqa.PSNR(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.SSIM:
        window_size = 3
        return lambda tensor_x, tensor_y:     piqa.SSIM(reduction="none", window_size=window_size, sigma=(window_size-1)/6, value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.MS_SSIM:
        window_size = 3
        return lambda tensor_x, tensor_y:  piqa.MS_SSIM(reduction="none", window_size=window_size, sigma=(window_size-1)/6, value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.VIFp:
        return lambda tensor_x, tensor_y:   piq.VIFLoss(reduction="none", data_range=1.0 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.FSIM:
        #return lambda tensor_x, tensor_y:     piqa.FSIM(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
        return lambda tensor_x, tensor_y:  piq.FSIMLoss(reduction="none", data_range=1.0 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.SR_SIM:
        return lambda tensor_x, tensor_y: piq.SRSIMLoss(reduction="none", scale=1, data_range=1.0 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.GMSD:
        return lambda tensor_x, tensor_y:     piqa.GMSD(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.MS_GMSD:
        return lambda tensor_x, tensor_y:  piqa.MS_GMSD(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.VSI:
        #return lambda tensor_x, tensor_y:      piqa.VSI(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
        return lambda tensor_x, tensor_y:   piq.VSILoss(reduction="none", data_range=1.0 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.DSS:
        return lambda tensor_x, tensor_y:   piq.DSSLoss(reduction="none", data_range=1.0 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.HaarPSI:
        return lambda tensor_x, tensor_y:  piqa.HaarPSI(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.MDSI:
        return lambda tensor_x, tensor_y:     piqa.MDSI(reduction="none", value_range=1.0).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.LPIPS:
        return lambda tensor_x, tensor_y:    piqa.LPIPS(reduction="none", network='vgg'  ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.PieAPP:
        return lambda tensor_x, tensor_y:    piq.PieAPP(reduction="none", data_range=1.0 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    elif metric == Metric_mode.DISTS:
        return lambda tensor_x, tensor_y:     piq.DISTS(reduction="none"                 ).to(device)(tensor_x.contiguous(), tensor_y.contiguous())
    else:
        raise Exception(f'Unsupported metric mode ({metric}).')

####################################################################################################

"""
Return width and height of images inside tensor
"""
def get_img_width_height(tensor: torch.Tensor):
    if len(tensor.shape) == 4: #batch x channel x width w height
        return tensor.shape[-2], tensor.shape[-1]
    else: #unknown
        raise Exception('Not Implemented')

####################################################################################################

"""
Return True if the metric is computable on tensor, based on image size in it, False otherwise.
"""
def is_metric_computable(metric: Metric_mode, tensor: torch.Tensor):
    img_width_height = get_img_width_height(tensor)
    if metric == Metric_mode.SSIM:
        if min(img_width_height) < 4:
            return False
    elif metric == Metric_mode.MS_SSIM:
        if min(img_width_height) < 64:
            return False
    elif metric == Metric_mode.VIFp:
        if min(img_width_height) < 41:
            return False
    elif metric == Metric_mode.FSIM:
        if min(img_width_height) < 2:
            return False
    elif metric == Metric_mode.SR_SIM:
        if min(img_width_height) < 16:
            return False
    elif metric == Metric_mode.MS_GMSD:
        if max(img_width_height) < 16:
            return False
    elif metric == Metric_mode.DSS:
        if img_width_height[0]*img_width_height[1] < 2**10: #e.g. smaller than 32x32, 16x64 or 8x128
            return False
    elif metric == Metric_mode.LPIPS:
        if min(img_width_height) < 16:
            return False
    elif metric == Metric_mode.PieAPP:
        if min(img_width_height) < 64:
            return False
    return True

####################################################################################################

"""
Compute given metric between torch.Tensor tensor_x and tensor_y.
"""
def compute_metric(tensor_x: torch.Tensor, tensor_y: torch.Tensor, metric: Metric_mode, device):
    assert tensor_x.shape == tensor_y.shape
    if not is_metric_computable(metric, tensor_x):
        return torch.full(torch.Size([tensor_x.shape[0]]), torch.nan)

    metric_fn = get_metric_fn(metric, device)
    return metric_fn(tensor_x, tensor_y)

####################################################################################################
##########          METRICS          ###############################################################
####################################################################################################

class Metric_fn(object):
    def __init__(self, metric: Metric_mode, device = torch.device("cpu")):
        self.metric = metric
        self.device = device
        self.metric_fn = get_metric_fn(self.metric, self.device)

    def __call__(self, tensor_x: torch.Tensor, tensor_y: torch.Tensor):
        assert tensor_x.shape == tensor_y.shape
        if not is_metric_computable(self.metric, tensor_x):
            return torch.full(torch.Size([tensor_x.shape[0]]), torch.nan)
        return self.metric_fn(tensor_x, tensor_y)

####################################################################################################
####################################################################################################
####################################################################################################