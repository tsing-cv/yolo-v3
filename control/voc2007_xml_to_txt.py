#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : voc2007_xml_to_txt.py
#   Author      : YunYang1994,tsing-cv
#   Created date: 2019-01-14 15:47:01
#   Description :
#
#================================================================

import os, sys 
sys.path.append("../")
print (os.getcwd())
import shutil
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET
from config import cfgs

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# sets=[('2012', 'train'), ('2012', 'val')]

parser = argparse.ArgumentParser()
parser.add_argument("--voc_path", default=cfgs.ROOT_DIR)
parser.add_argument("--dataset_info_path", default="{}/dataset".format(cfgs.ROOT_DIR))
flags = parser.parse_args()


def convert_annotation(year, image_id, list_file):
    xml_path = os.path.join(flags.voc_path, 'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in cfgs.label_num_map or int(difficult)==1:
            continue
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(",".join([str(a) for a in b]) + "," + cls+"\n")



for year, image_set in sets:
    text_path = os.path.join(flags.voc_path, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set))
    if not os.path.exists(text_path): continue
    image_ids = open(text_path).read().strip().split()


    annot_path = os.path.join(flags.dataset_info_path, image_set, "ANNOT")
    jpeg_path = os.path.join(flags.dataset_info_path, image_set, "JPEG")
    print ("\n\n{}\nXML data is converting >>>\n\tAnnotations will be saved in {}\n\tImages will be saved in {}".format(
        "@@@"*20, annot_path, jpeg_path))
    if not os.path.exists(annot_path):
        os.makedirs(annot_path)
    if not os.path.exists(jpeg_path):
        os.makedirs(jpeg_path)

    pbar = tqdm(image_ids, total=len(image_ids), ncols=120)
    for image_id in pbar:
        image_path = os.path.join(flags.voc_path, 'VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
        
        shutil.copy(image_path, jpeg_path)
        pbar.set_description("\t{}".format(image_path))
        with open(os.path.join(annot_path, "{}.txt".format(image_id)), 'w') as f:
            convert_annotation(year, image_id, f)
    print ("\nConverted {}_{} !!!\n{}".format(image_set, year, "@@@"*20))


