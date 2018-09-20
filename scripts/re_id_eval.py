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

# VISUALIZE = True
VISUALIZE = False
VISUALIZE_RE_ID = True

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
    det_cls = np.zeros(len(det_bb))

    iou_array = iou(det_bb, gt_bb)
    pos_ind = np.where(np.amax(iou_array, axis=1) >= iou_thresh)
    if not len(pos_ind[0]):
        print('Warning: No matching bb!')
    iou_array = iou_array[pos_ind]
    pos_det_cls = gt_cls[np.argmax(iou_array, axis=1)]
    det_cls[pos_ind] = pos_det_cls
    
    return det_cls, pos_ind[0]

def viz_detection(args, fname, bb_list, cls_list=None, gt_cls_list=None):
    input_file = os.path.join(args.anno_dir, '..', 'frames', os.path.basename(fname).split('.')[0] + '.jpg')
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    if cls_list.size:
        final = draw_final_outputs(img, bb_list, bb_list_input=True, cls_list=cls_list, gt_cls_list=gt_cls_list)
    else:
        final = draw_final_outputs(img, bb_list, tags_on=False, bb_list_input=True)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite(os.path.join('output','test03',os.path.basename(input_file)), viz)
    # tpviz.interactive_imshow(viz)

def re_id_viz_detection(args, query_fname, gallery_fname, query_bb, top1_bb):
    query_file = os.path.join(args.anno_dir, '..', 'frames', os.path.basename(query_fname).split('.')[0] + '.jpg')
    gallery_file = os.path.join(args.anno_dir, '..', 'frames', os.path.basename(gallery_fname).split('.')[0] + '.jpg')
    query_image = cv2.imread(query_file, cv2.IMREAD_COLOR)
    gallery_img = cv2.imread(gallery_file, cv2.IMREAD_COLOR)
    query_image_with_bb = draw_final_outputs(query_image, query_bb, tags_on=False, bb_list_input=True)
    gallery_image_with_bb = draw_final_outputs(gallery_img, [top1_bb], tags_on=False, bb_list_input=True)
    viz = np.concatenate((query_image_with_bb, gallery_image_with_bb), axis=1)
    cv2.imwrite(os.path.join('output','test04',os.path.basename(query_fname)), viz)
    # tpviz.interactive_imshow(viz)

def re_id_eval(args):
    with open(args.gallery_file, 'r') as gallery_file:
        gallery_list = json.load(gallery_file)
        gallery_bb = []
        gallery_fname = []
        for frame in gallery_list:
            """
                frame[0]: abs fname
                frame[1]: bb list
                frame[2]: label list
                frame[3]: score list
                frame[4]: feature vectors list
            """
            if VISUALIZE:
                viz_detection(args, frame[0], frame[1])

            gt_bb_array, gt_cls_array = read_annotations(
                os.path.join(args.anno_dir, os.path.basename(frame[0]).split('.')[0] + '.txt'))
            if not frame[1]:
                # print('No detection')
                continue
            det_gt_cls_array, pos_ind = bb_cls_matching(np.array(frame[1]), gt_bb_array, gt_cls_array)
            for bb, fv, det_cls in zip(frame[1], frame[4], det_gt_cls_array):
                gallery_bb.append(fv + bb + [det_cls])
                # corresponding images
                gallery_fname.append(frame[0])
    gallery_bb = np.array(gallery_bb)
    # to see how many dets are generated
    print(gallery_bb.shape)
            
    with open(args.query_file, 'r') as query_file:
        query_list = json.load(query_file)
        tp_top20 = 0.0
        # with tqdm.tqdm(total=len(query_list), **get_tqdm_kwargs()) as tqdm_bar:
        #     for query in query_list:
        #         """
        #             query[0]: query id list
        #             query[1]: feature vector list
        #         """
        #         fv = np.array(query[1][0][0])
        #         # print(fv.size)
        #         distance = []
        #         for gfv in gallery_bb:
        #             distance.append(np.linalg.norm(fv - gfv[:args.fv_length])) # 256 - fv length
        #         distance_array = np.array(distance)
        #         index_sort = np.argsort(distance_array)
        #         cls_top20 = gallery_bb[index_sort[:20], -1]
        #         if query[0][0] in cls_top20.astype(int).tolist():
        #             print(query[0][0], cls_top20.astype(int).tolist())
        #             # print('yay')
        #             tp_top20 += 1
        #         tqdm_bar.update(1)
        for query in query_list:
            """
                query[0]: query fname
                query[1]: query id list
                query[2]: feature vector list
                query[3]: original box list
            """
            fv = np.array(query[2][0][0])
            # print(fv.size)
            distance = []
            for gfv in gallery_bb:
                distance.append(np.linalg.norm(fv - gfv[:args.fv_length])) # 256 - fv length
            distance_array = np.array(distance)
            index_sort = np.argsort(distance_array)
            cls_top20 = gallery_bb[index_sort[:20], -1]
            top = 1 # top2
            top1_image = gallery_fname[index_sort[top]]

            if VISUALIZE_RE_ID:
                re_id_viz_detection(args, query[0], top1_image, query[3][0], gallery_bb[index_sort[top], -5:-1])

            if query[1][0] in cls_top20.astype(int).tolist():
                print(query[1][0], cls_top20.astype(int).tolist())
                # print('yay')
                tp_top20 += 1
    
    print('Top 20 accuracy: ' + str(tp_top20/len(query_list)))

def classifier_eval(agrs):
    with open(args.classification_file, 'r') as cls_file:
        classification_list = json.load(cls_file)

    # with tqdm.tqdm(total=len(classification_list), **get_tqdm_kwargs()) as tqdm_bar:
    num_tp_classification = 0
    num_tp_det = 0
    num_gt = 0
    num_det = 0
    for result in classification_list:
    # adding range when debugging
        """
            result[0]: fname
            result[1]: bb_list
            result[2]: prob_list
        """
        # viz_detection(args, result[0], result[1])

        gt_bb_array, gt_cls_array = read_annotations(
            os.path.join(args.anno_dir, os.path.basename(result[0]).split('.')[0] + '.txt'))
        # throw away unidentified persons
        gt_bb_array = gt_bb_array[gt_cls_array != -2]
        gt_cls_array = gt_cls_array[gt_cls_array != -2]
        if not len(gt_bb_array):
            print('Frame with no identified pede encountered!')
            continue
        if not len(result[1]):
            continue
        det_gt_cls_array, pos_ind = bb_cls_matching(np.array(result[1]), gt_bb_array, gt_cls_array, iou_thresh=0.7)
        det_cls_array = np.argmax(np.array(result[2]), axis=1)

        if VISUALIZE:
            # the zeros here mean that the detection doesn't match to any id
            det_gt_cls_array_all = np.zeros(len(det_cls_array), dtype=int)
            det_gt_cls_array_all[pos_ind] = det_gt_cls_array[pos_ind]
            viz_detection(args, result[0], result[1], det_cls_array, det_gt_cls_array_all)

        # print(det_cls_array)
        # here we only consider the iou thresholded ones, this applies to gts as well
        # print(pos_ind)
        # print(det_cls_array[pos_ind])
        # print(det_gt_cls_array[pos_ind])
        # print('\n')
        num_tp_classification += len(np.where(det_cls_array[pos_ind] == det_gt_cls_array[pos_ind])[0])
        # is it correct to only eval classification on valid dets?
        num_tp_det += len(pos_ind)
        num_gt += len(det_gt_cls_array)
        num_det += len(det_cls_array)
    print(num_tp_classification)
    print(num_tp_det)
    print(num_gt)
    print(num_det)
    print('Classification accuracy: ' + str(float(num_tp_classification) / num_det))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file')
    parser.add_argument('--gallery_file')
    parser.add_argument('--classification_file')
    parser.add_argument('--fv_length', default=2048)
    parser.add_argument('--anno_dir', default='/media/yingges/TOSHIBA EXT/datasets/re-ID/PRW-v16.04.20/converted_annotations02')
    args = parser.parse_args()

    if args.query_file and args.gallery_file:
        re_id_eval(args)
    elif args.classification_file:
        classifier_eval(args)

        