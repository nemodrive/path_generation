from . import train

TRAINERS = {
    "TrainDefault": train.TrainDefault,
}


def get_train(cfg, *args, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in TRAINERS,\
        "Please provide a valid trainer name."
    return TRAINERS[cfg.name](cfg, *args, **kwargs)
