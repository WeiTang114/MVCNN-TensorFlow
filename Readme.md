This is a simple implementation of Multi-View CNN (MVCNN) introduced by Su et al. for 3D shape recognition, with TensorFlow. The original paper is here: [Multi-view Convolutional Neural Networks for 3D Shape Recognition](http://vis-www.cs.umass.edu/mvcnn/).

A specific model is implemented:

1. AlexNet
2. 12-view
3. view-pooling after pool5 layer

# Requirements

- CUDA (7.5 up better)
- TensorFlow (0.10 up better)
- python 2.7
- Some other python packages


# Usage

## Data preparation

You need to prepare the rendered views data for 3D shapes. We use ModelNet40 for example. Every 3D shape is rendered in phong-shading with no texture, in white color over black background, elevation of 30 degree, and 12 azimuths every 30 degree. 

Every 3D shape is defined in a text file. See data\_sample/view/list/train/airplane/airplane_0001.off.txt for example. The first line is category id, starting from 0. The second line is number of views, default 12. And the following 12 lines are image files of the 12 views.

You need 3 list files for training, validation, and testing. See data\_sample/view/train_lists.txt for example. The second column is category id. Please specify the paths of the list files in **globals.py**.

## Pretrained model preparation

The pretrained AlexNet model is split to 3 files because Github has a limitation of 100MB/file. To merge them, please run

```
$ ./prepare_pretrained_alexnet.sh
```

## Training

To train at the first time, run

```
$ mkdir tmp
$ python train.py --train_dir=`pwd`/tmp --caffemodel=`pwd`/alexnet_imagenet.npy --learning_rate=0.0001
```

To fine-tune, run

```
# N is your checkpoint iteration
$ python train.py --train_dir=`pwd`/tmp --weights=`pwd`/tmp/model.ckpt-N --learning_rate=0.0001
```

## Testing

```
# N is your checkpoint iteration
$ python test.py --weights=`pwd`/tmp/model.ckpt-N
```

# License
MIT
