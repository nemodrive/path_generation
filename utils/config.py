from argparse import Namespace
from termcolor import colored as clr
import os
import yaml
from time import time
from typing import List


def dict_to_namespace(dct: dict):
    namespace = Namespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config_file',
        default='configs/default_base.yaml',
        nargs="+",
        dest='config_file',
        help='Configuration file.'
    )
    return arg_parser.parse_args()


def read_config():
    args = parse_args()
    d = yaml.load(open(args.config_file))
    n = dict_to_namespace(d)
    return n


def add_to_cfg(cfg: Namespace, subgroups: List[str], new_arg: str, new_arg_value) -> None:
    # Add to list of attributes defined by subgroups new_arg = new_arg_value

    if len(subgroups) > 0:
        for arg in subgroups:
            if hasattr(cfg, arg):
                setattr(getattr(cfg, arg), new_arg, new_arg_value)
    else:
        for arg in cfg.__dict__.keys():
            if isinstance(getattr(cfg, arg), Namespace):
                setattr(getattr(cfg, arg), new_arg, new_arg_value)

