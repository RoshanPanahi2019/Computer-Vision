import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

try:
    from . import resnet
except:
    import resnet
    
class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.query = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.key = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.value = nn.Conv1d(in_channel, in_channel, 1)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out


class EventNet(nn.Module):
    def __init__(self, model_name, num_classes, patch_size=64, use_feature_maps=True, use_self_attn=False, modified=False):

        super().__init__()
        self.use_feature_maps = use_feature_maps
        self.use_self_attn = use_self_attn
        
        self.base1 = getattr(resnet, model_name)(pretrained=True, modified=modified)

        self.base2 = getattr(resnet, model_name)(pretrained=True, modified=modified)
        if use_feature_maps:
            self.final_layer = nn.Conv2d(2048 * 2, num_classes, patch_size//8)
        else:
            self.final_layer = nn.Linear(2048 * 2, num_classes)

        if use_self_attn:
            self.self_attn = SelfAttention(2048)

    def extract(self, bbx, cbx):
        bbx, bbx_f = self.base1(bbx)
        cbx, cbx_f = self.base2(cbx)

        if self.use_self_attn:
            bbx_f = self.self_attn(bbx_f)
            cbx_f = self.self_attn(cbx_f)

        x = torch.cat([bbx_f, cbx_f], 1)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        return x      
    
    
    def forward(self, bbx, cbx):
        bbx, bbx_f = self.base1(bbx)
        cbx, cbx_f = self.base2(cbx)

        if self.use_self_attn:
            bbx_f = self.self_attn(bbx_f)
            cbx_f = self.self_attn(cbx_f)

        xf = torch.cat([bbx_f, cbx_f], 1)
        
        if not self.use_feature_maps:
            xf = F.adaptive_avg_pool2d(xf, 1)
            xf = xf.view(xf.size(0), -1)

        x = self.final_layer(xf).view(xf.size(0), -1)

        return x, xf