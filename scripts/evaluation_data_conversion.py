import argparse
import os
import scipy.io
from os.path import join as pjoin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/yingges/Desktop/Research/thesis01/PRW-v16.04.20')
    parser.add_argument('--output_dir_postfix', default='')
    parser.add_argument('--set', default='test')
    args = parser.parse_args()

    frame_list_mat = scipy.io.loadmat(pjoin(args.data_dir, 'frame_' + args.set + '.mat'))
    frame_list = frame_list_mat['img_index_' + args.set]

    try:
        output_dir = pjoin(args.data_dir, args.set + '_converted_annotations' + args.output_dir_postfix)
        os.mkdir(output_dir)
    except OSError:
        print('Output folder already exists.')
    for idx, frame in enumerate(frame_list):
        anno_data = scipy.io.loadmat(pjoin(args.data_dir, 'annotations', frame[0][0] + '.jpg.mat'))

        # print(anno_data)
        if 'box_new' in anno_data:
            gt_bb = anno_data['box_new']
        elif 'anno_file' in anno_data:
            gt_bb = anno_data['anno_file']
        elif 'anno_previous' in anno_data:
            gt_bb = anno_data['anno_previous']
        else:
            raise Exception(frame[0][0] + ' bounding boxes info missing!')

        include_all = True
        if not include_all:
            if len(gt_bb[gt_bb[:,0] != -2]) == 0:
                print('Skiping a frame w/o fg objects...')
                continue

        gt_bb[:,3] = gt_bb[:,1] + gt_bb[:,3]
        gt_bb[:,4] = gt_bb[:,2] + gt_bb[:,4]
        gt_bb = gt_bb.astype(str)
        with open(pjoin(output_dir, frame[0][0] + '.txt'), 'w') as f:
            for bb in gt_bb:
                if include_all:
                    f.write(' '.join(['pedestrian', bb[1], bb[2], bb[3], bb[4], bb[0]]) + '\n')
                    # f.write(' '.join(['pedestrian', bb[1], bb[2], bb[3], bb[4], str(int(bb[0]))]) + '\n')
                # add id - bb[0]
                # else:
                #     if bb[0] == -2:
                #         f.write(' '.join(['BG', bb[1], bb[2], bb[3], bb[4]]) + '\n')
                #     else:
                #         f.write(' '.join(['pedestrian', bb[1], bb[2], bb[3], bb[4]]) + '\n')

