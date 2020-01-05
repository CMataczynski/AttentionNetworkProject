import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from model import ResidualAttentionModel_56, ResidualAttentionModel_92
from torchvision.transforms import functional as tf
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

class GradCam:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.fmap_pool = dict()
        self.grad_pool = dict()

        def forward_hook(module, input, output):
            self.fmap_pool[module] = output.detach().cpu()

        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[module] = grad_out[0].detach().cpu()

        for layer in layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        assert layer in self.layers, f'{layer} not in {self.layers}'
        fmap_b = self.fmap_pool[layer]  # [N, C, fmpH, fmpW]
        grad_b = self.grad_pool[layer]  # [N, C, fmpH, fmpW]

        grad_b = F.adaptive_avg_pool2d(grad_b, (1, 1))  # [N, C, 1, 1]
        gcam_b = (fmap_b * grad_b).sum(dim=1, keepdim=True)  # [N, 1, fmpH, fmpW]
        gcam_b = F.relu(gcam_b)

        return gcam_b


class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        return layer.grad.cpu()

def colorize(tensor, colormap=plt.cm.get_cmap('viridis')):
    '''Apply colormap to tensor
    Args:
        tensor: (FloatTensor), sized [N, 1, H, W]
        colormap: (plt.cm.*)
    Return:
        tensor: (FloatTensor), sized [N, 3, H, W]
    '''
    tensor = tensor.clamp(min=0.0)
    tensor = tensor.squeeze(dim=1).numpy() # [N, H, W]
    tensor = colormap(tensor)[..., :3] # [N, H, W, 3]
    tensor = torch.from_numpy(tensor).float()
    print(tensor.shape)
    tensor = normalize(tensor,pil=True)
    print(tensor.shape)
    tensor = tensor.permute(0, 3, 1, 2) # [N, 3, H, W]
    print(tensor.shape)
    return tensor

def normalize(tensor, eps=1e-12, pil=False):
    '''Normalize each tensor in mini-batch like Min-Max Scaler
    Args:
        tensor: (FloatTensor), sized [N, C, H, W]
    Return:
        tensor: (FloatTensor) ranged [0, 1], sized [N, C, H, W]
    '''
    N = tensor.size(0)
    min_val = tensor.contiguous().view(N, -1).min(dim=1)[0]
    tensor = tensor - min_val.view(N, 1, 1, 1)
    max_val = tensor.contiguous().view(N, -1).max(dim=1)[0]
    if not pil:
        tensor = tensor / ((max_val + eps).view(N, 1, 1, 1))
    else:
        tensor = 255 * tensor / ((max_val + eps).view(N, 1, 1, 1))
        tensor = tensor.to(torch.uint8)
    return tensor


class Predictor:

    def __init__(self):
        seed = 999
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = ResidualAttentionModel_56()
        self.model_path = "./Res_56_new.pth"
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()

        self.testing_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        self.prediction = None
        self.gc_img = None


    def predict(self, in_img):

        img = in_img
        img = self.testing_transforms(img)

        inp_b = img.to(self.device)
        self.img = in_img
        #self.gc_img =in_img
        # 243: boxer
        # 283: tiger cat
        # grad_b = torch.zeros_like(out_b, device=device)
        # grad_b[:, out_b.argmax(dim=1)] = +1.0
        # out_b.backward(gradient=grad_b)
        with torch.no_grad():
            inputs = inp_b
            ncrops, c, h, w = inputs.size()
            result = self.model(inputs.view(-1, c, h, w))
            result_avg = result.view(1, ncrops, -1).mean(1)
            self.probs = result_avg

        with GradCam(self.model, [self.model.features]) as gcam:
            ncrops, c, h, w = inp_b.size()
            result = self.model(inp_b.view(-1, c, h, w))
            out_b = gcam(inp_b)  # [N, C]
            # print(out_b[:,0])
            out_b[:, 0].mean().backward()

            gcam_b = gcam.get(self.model.features)  # [N, 1, fmpH, fmpW]
            gcam_b = F.interpolate(gcam_b, [224, 224], mode='bilinear', align_corners=False)  # [N, 1, inpH, inpW]
            #print(gcam_b.max()-gcam_b.min())
            self.gacm_img = normalize(gcam_b)

        # with GuidedBackPropogation(self.model) as gdbp:
        #     ncrops, c, h, w = inp_b.size()
        #     inp_b = inp_b.requires_grad_()  # Enable recording inp_b's gradient
        #     out_b = gdbp(inp_b)
        #     out_b[:, 0].max().backward()
        #
        #     grad_b = gdbp.get(inp_b)  # [N, 3, inpH, inpW]
        #     grad_b = grad_b.mean(dim=1, keepdim=True)  # [N, 1, inpH, inpW]
        #     #print(grad_b.max()-grad_b.min())
        #     self.gradb_img = normalize(grad_b)
        #
        # mixed = gcam_b * grad_b
        # self.mixed_img = normalize(mixed)


    def getGradcamImage(self):
        changed = np.copy(self.img) #H,W,C
        size = 224
        x_min = [0, changed.shape[1]-size,0, changed.shape[1]-size, int((changed.shape[1]-size)/2)]
        y_min = [0,0,changed.shape[0]-size, changed.shape[0]-size, int((changed.shape[0]-size)/2)]
        heatmap = self.gacm_img
        heatmap_overlay = np.zeros(changed[:,:,0].shape)
        for i in range(heatmap.shape[0]):
            heatmap_i = heatmap[i,0,:,:]
            #img = changed[y_min[i]:y_min[i]+size,x_min[i]:x_min[i]+size,:]
            #print(heatmap_i.max()-heatmap_i.min())
            #img = tf.to_pil_image(img)  # assuming your image in x
            heatmap_overlay[y_min[i]:y_min[i] + size, x_min[i]:x_min[i] + size] += np.rot90(heatmap_i,k=0)
            heatmap_overlay[y_min[i]:y_min[i] + size, x_min[i]:x_min[i] + size] /= 2
        print(heatmap_overlay.shape)
        heatmap_overlay = torch.tensor(heatmap_overlay).view(1, 1,heatmap_overlay.shape[0],heatmap_overlay.shape[1])
        heatmap_overlay = normalize(heatmap_overlay)
        print(heatmap_overlay.max())
        heatmap_overlay = colorize(heatmap_overlay)
        print(heatmap_overlay.shape)
        heatmap_overlay = heatmap_overlay.view(3,heatmap_overlay.shape[2],heatmap_overlay.shape[3])
        print(heatmap_overlay.shape)
        heatmap_overlay = heatmap_overlay.permute(1,2,0).to(torch.uint8).numpy()
        print(heatmap_overlay.shape)
        heatmap_overlay = tf.to_pil_image(heatmap_overlay)
        heatmap_overlay = heatmap_overlay.filter(ImageFilter.GaussianBlur(radius=7))

        print(changed.shape)
        img = tf.to_pil_image(changed)
        changed = Image.blend(img,heatmap_overlay,0.5)
        # changed = heatmap[4,:,:,:].permute(1,2,0).to(torch.uint8).numpy()
        # print(changed.shape)
        #changed = np.array(self.img)[y_min[1]:y_min[1] + size, x_min[1]:x_min[1] + size,:]
        return np.array(changed)

    def getPrediction(self):
        return self.probs.data.numpy()[0][0] * 100
