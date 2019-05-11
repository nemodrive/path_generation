from . import base
from . import unet
from . import depth_wild
from . import flow_net_s
from . import flow_net_c

MODELS = {
    "BaseModel": base.BaseModel,
    "DepthWild": depth_wild.DepthWild,
    "UNet": unet.UNet,
    "FlowNet2S": flow_net_s.FlowNet2S,
    "FlowNet2C": flow_net_c.FlowNet2C,
}


def get_model(cfg, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, **kwargs)
