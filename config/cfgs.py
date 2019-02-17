#! /usr/bin/env python3
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : cfgs.py
#   Author      : tsing-cv
#   Created date: 2019-02-15 18:41:35
#   Description :
#
#================================================================

# edition *******************************************************
DATASET = 'voc'
TIME    = 20190214

# dataset *******************************************************
IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE       = 16
EPOCHS           = 2500
learning_rate    = 0.001 # if Nan, set 0.0005, 0.0001
DECAY_STEPS      = 100
DECAY_RATE       = 0.94
SHUFFLE_SIZE     = 200

ROOT_DIR         = "C:/Users/tsing/yolo-v3"
train_tfrecord   = "{}/dataset/TFRECORD/{}_train.tfrecords".format(ROOT_DIR, DATASET)
test_tfrecord    = "{}/dataset/TFRECORD/{}_valid.tfrecords".format(ROOT_DIR, DATASET)

# train *********************************************************
gpus                 = (0,)
moving_average_decay = 0.99
max_number_of_steps  = 1e+6
gpu_memory_fraction  = -1
using_moving_average = True
checkpoint_path      = ""
log_every_n_steps    = 20
eval_interval        = 100
checkpoint_path      = "{}/outputs/{}{}".format(ROOT_DIR, DATASET, TIME)



# dataset_map ***************************************************
if DATASET == "voc":
    label_num_map   = {
                        "aeroplane": 0,
                        "bicycle": 1,
                        "bird": 2,
                        "boat": 3,
                        "bottle": 4,
                        "bus": 5,
                        "car": 6,
                        "cat": 7,
                        "chair": 8,
                        "cow": 9,
                        "diningtable": 10,
                        "dog": 11,
                        "horse": 12,
                        "motorbike": 13,
                        "person": 14,
                        "pottedplant": 15,
                        "sheep": 16,
                        "sofa": 17,
                        "train": 18,
                        "tvmonitor": 19
                        }
    ANCHORS         =  [[0.057692307692307696, 0.08173076923076923],
                        [0.11057692307692307, 0.20192307692307693],
                        [0.16346153846153846, 0.44471153846153844],
                        [0.27884615384615385, 0.6875],
                        [0.2932692307692308, 0.23317307692307693],
                        [0.4110576923076923, 0.4326923076923077],
                        [0.514423076923076, 0.7860576923076923],
                        [0.7836538461538461, 0.4639423076923077],
                        [0.8629807692307693, 0.8629807692307693],
                        [IMAGE_H, IMAGE_W]]
    CLASSES          = {v:k for k,v in label_num_map.items()}
    NUM_CLASSES      = len(CLASSES)