from argparse import Namespace
from termcolor import colored as clr
import os
import yaml
#from logbook import Logger
from time import time

def dict_to_namespace(dct):
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
    print('hhhhhhhhhhhhhhhh', d)
    n = dict_to_namespace(d)
    return n
