from . import base

MODELS = {
    "BaseModel": base.BaseModel,
}


def get_model(cfg, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, **kwargs)
