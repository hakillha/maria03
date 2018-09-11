import argparse
import cv2
import json
import numpy as np
import os
import scipy
import sys
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack.utils.utils import get_tqdm_kwargs

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.np_box_ops import iou
from viz import draw_final_outputs


def read_annotations(fname):
    gt_bb = []
    gt_cls = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.split()
            gt_bb.append(list(map(float,line[1:5])))
            gt_cls.append(int(float(line[5])))
    gt_bb = np.array(gt_bb, dtype=np.float32)
    gt_cls = np.array(gt_cls, dtype=np.int16)

    return gt_bb, gt_cls

def bb_cls_matching(det_bb, gt_bb, gt_cls, iou_thresh=0.5):
    iou_array = iou(det_bb, gt_bb)
    det_cls = np.zeros(len(iou_array))
    pos_ind = np.where(np.amax(iou_array, axis=1) >= iou_thresh)
    if not len(pos_ind[0]):
        print('Warning: No matching bb!')
    iou_array = iou_array[pos_ind]
    pos_det_cls = gt_cls[np.argmax(iou_array, axis=1)]
    det_cls[pos_ind] = pos_det_cls
    
    return det_cls, pos_ind

def viz_detection(args, fname, bb_list):
    input_file = os.path.join(args.anno_dir, '..', 'frames', os.path.basename(fname).split('.')[0] + '.jpg')
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    # print(result[1])
    final = draw_final_outputs(img, bb_list, tags_on=False, bb_list_input=True)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite(os.path.basename(input_file), viz)
    # tpviz.interactive_imshow(viz)

def re_id_eval(args):
    with open(args.gallery_file, 'r') as gallery_file:
        gallery_list = json.load(gallery_file)
        gallery_bb = []
        for frame in gallery_list:
            """
                frame[0]: abs fname
                frame[1]: bb list
                frame[2]: label list
                frame[3]: score list
                frame[4]: feature vectors list
            """
            viz_detection(args, frame[0], frame[1])

            gt_bb_array, gt_cls_array = read_annotations(
                os.path.join(args.anno_dir, os.path.basename(frame[0]).split('.')[0] + '.txt'))
            if not frame[1]:
                # print('No detection')
                continue
            det_gt_cls_array, pos_ind = bb_cls_matching(np.array(frame[1]), gt_bb_array, gt_cls_array)
            for bb, fv, det_cls in zip(frame[1], frame[4], det_gt_cls_array):
                gallery_bb.append(fv + bb + [det_cls]) 
    gallery_bb = np.array(gallery_bb)
    print(gallery_bb.shape)
            
    with open(args.query_file, 'r') as query_file:
        query_list = json.load(query_file)
        tp_top20 = 0.0
        with tqdm.tqdm(total=len(query_list), **get_tqdm_kwargs()) as tqdm_bar:
            for query in query_list:
                """
                    query[0]: feature vector list
                    query[1]: query id list
                """
                # print(len(query[0][0][0]))
                fv = np.array(query[1][0][0])
                # fv = query[0][0][0]
                distance = []
                for gfv in gallery_bb:
                    distance.append(np.linalg.norm(fv - gfv[:256]))
                distance_array = np.array(distance)
                index_sort = np.argsort(distance_array)
                cls_top20 = gallery_bb[index_sort[:20], 260]
                if query[0][0] in cls_top20.astype(int).tolist():
                    # print(query[0][0], cls_top20.astype(int).tolist())
                    print('yay')
                    tp_top20 += 1
                tqdm_bar.update(1)
    
    print('Top 20 accuracy: ' + str(tp_top20/len(query_list)))

def classifier_eval(agrs):
    with open(args.classification_file, 'r') as cls_file:
        classification_list = json.load(cls_file)

    # with tqdm.tqdm(total=len(classification_list), **get_tqdm_kwargs()) as tqdm_bar:
    num_tp = 0
    num_gt = 0
    for result in classification_list:
        """
            result[0]: fname
            result[1]: bb_list
            result[2]: prob_list
        """
        # viz_detection(args, result[0], result[1])

        gt_bb_array, gt_cls_array = read_annotations(
            os.path.join(args.anno_dir, os.path.basename(result[0]).split('.')[0] + '.txt'))
        print(np.array(result[1]))
        print(gt_bb_array)
        if not len(result[1]):
            continue
        det_gt_cls_array, pos_ind = bb_cls_matching(np.array(result[1]), gt_bb_array, gt_cls_array, iou_thresh=0.7)
        # print(len(result[1]))
        # print(str(len(gt_bb_array)) + '\n')
        # print(pos_ind[0])
        det_cls_array = np.argmax(np.array(result[2]), axis=1)
        # here we only consider the iou thresholded ones, this applies to gts as well
        num_tp += len(np.where(det_cls_array[pos_ind] == det_gt_cls_array[pos_ind]))
        num_gt += len(pos_ind)
    # print(num_gt)
    print('Classification accuracy: ' + str(float(num_tp) / num_gt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file')
    parser.add_argument('--gallery_file')
    parser.add_argument('--classification_file')
    parser.add_argument('--anno_dir', default='/media/yingges/TOSHIBA EXT/datasets/re-ID/PRW-v16.04.20/converted_annotations02')
    args = parser.parse_args()

    if args.query_file and args.gallery_file:
        re_id_eval(args)
    elif args.classification_file:
        classifier_eval(args)

        