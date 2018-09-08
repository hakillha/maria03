import argparse
import json
import numpy as np
import os
import scipy
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.np_box_ops import iou


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

def bb_cls_matching(det_bb, gt_bb, gt_cls):
    iou_array = iou(det_bb, gt_bb)
    det_cls = np.zeros(len(iou_array))
    pos_ind = np.where(np.amax(iou_array, axis=1) >= 0.5)
    iou_array = iou_array[pos_ind]
    pos_det_cls = gt_cls[np.argmax(iou_array, axis=1)]
    det_cls[pos_ind] = pos_det_cls
    
    return det_cls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file')
    parser.add_argument('--gallery_file')
    parser.add_argument('--anno_dir', default='/media/yingges/TOSHIBA EXT/datasets/re-ID/PRW-v16.04.20/converted_annotations02')
    args = parser.parse_args()

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
            gt_bb_array, gt_cls_array = read_annotations(
                os.path.join(args.anno_dir, os.path.basename(frame[0]).split('.')[0] + '.txt'))
            det_cls_array = bb_cls_matching(np.array(frame[1]), gt_bb_array, gt_cls_array)
            for bb, fv, det_cls in zip(frame[1], frame[4], det_cls_array):
                gallery_bb.append(fv + bb + [det_cls]) 
    gallery_bb = np.array(gallery_bb)
            
    with open(args.query_file, 'r') as query_file:
        query_list = json.load(query_file)
        tp_top20 = 0.0
        for query_id, query in enumerate(query_list):
            """
                query[0]: feature vector list
                query[1]: query id list
            """
            # print(len(query[0][0][0]))
            fv = np.array(query[0][0][0])
            # fv = query[0][0][0]
            distance = []
            print(query_id)
            for gfv in gallery_bb:
                distance.append(np.linalg.norm(fv - gfv[:256]))
                distance = np.array(distance)
                index_sort = np.argsort(distance)
                cls_top20 = gallery_bb[index_sort[:20], 260]
                if query[1][0] in cls_top20.tolist():
                    tp_top20 += 1
        print('Top 20 accuracy: ' + str(tp_top20/len(query_list)))

        