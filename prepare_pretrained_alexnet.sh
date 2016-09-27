#!/bin/bash
# This is to merge split alexnet pretrained model files to a .npy file.
# They are split because the original file is too large for Github's 100MB limit.

ROOT=`dirname $0`
cat $ROOT/pretrained_model/alexnet_imagenet.npy.split.a{a..c} > $ROOT/alexnet_imagenet.npy
