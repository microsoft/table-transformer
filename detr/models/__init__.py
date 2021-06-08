# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_multi import build as build_multi


def build_model(args):
    return build(args)

def build_model_multi(args):
    return build_multi(args)