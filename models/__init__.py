from .ProxyNet import *
from .Net import ProxyNet_sub, AdjustEncoder
from .UPCB import *
from .model_keypoints import *
from .KeypointNet import Encoder
from .PCB import PCB, ResNet50_self, ResNet50_anchor


__factory = {
    'proxy': ProxyNet,
    'upcb':UPCB,
    'distill': DisTill_unit,
    'pcb': PCB,
    'resnet': ResNet50_self,
    'proxy_sub': ProxyNet_sub
}


def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)