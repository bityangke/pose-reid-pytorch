# coding=utf-8
# Created by run on 2019-03-11 20:49
# Copyright © 2019 Alan. All rights reserved.

import time
import argparse
import torch
import cv2
import os
import numpy as np
import yaml
from src.pose.openpose import *
from src.reid.svd import *
from utils import *
from collections import defaultdict,OrderedDict
import json

parser = argparse.ArgumentParser(description='Pose Re-identification System')
parser.add_argument('--gpu_ids',default='1',type=str,help='run on specific gpus, e.g. gpu_ids: 0,1,2')
parser.add_argument('--plot_skeleton',default=True,type=bool,help='whether plot the skeleton')
parser.add_argument('--svd_val',default=2,type=float,help='svd vectors valve')
parser.add_argument('--pose_val',default=14,type=int,help='pose selection valve')
parser.add_argument('--src_video_path',default='./zhaobenshan.mkv',type=str,help='source video path')
parser.add_argument('--dst_video_path',default='./benshan.avi',type=str,help='output video path')
parser.add_argument('--dst_pic_path',default='./frameResult',type=str,help='picture save path')
parser.add_argument('--json_path',default='./openpose_id.json',type=str,help='store the joint information')
parser.add_argument('--txt_path',default='./result_output.txt',type=str,help='result log')


def set_gpu_ids(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids


def load_openpose(model_src = './models/pose_model.pth'):
    print('loading Openpose...')
    pose_model = get_model('vgg19').cuda()
    pose_model.load_state_dict(torch.load(model_src))
    pose_model = torch.nn.DataParallel(pose_model).cuda()
    pose_model.float()
    pose_model.eval()
    return pose_model


def load_reid(path = './models/ft_ResNet50/'):
    print('loading SVDNet...')
    config_path = path + 'opts.yaml'

    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    use_dense = config['use_dense']

    if use_dense:
        id_model_structure = ft_net_dense(675)
    else:
        id_model_structure = ft_net(675)

    model_path = path + 'svd_model.pth'

    id_model = load_network(id_model_structure, model_path)

    id_model.model.fc = torch.nn.Sequential()
    id_model.classifier = torch.nn.Sequential()

    id_model = id_model.eval()
    id_model = id_model.cuda()

    return id_model


if __name__ == '__main__':
    opt = parser.parse_args()
    gpu_ids = opt.gpu_ids
    plot_skeleton = opt.plot_skeleton
    svd_val = opt.svd_val
    src_video_path = opt.src_video_path
    dst_video_path = opt.dst_video_path
    dst_pic_path = opt.dst_pic_path
    pose_val = opt.pose_val
    json_path = opt.json_path
    txt_path = opt.txt_path

    try:
        os.remove(txt_path)
    except OSError:
        pass
    f = open(txt_path, 'a')

    # read the video
    vc = cv2.VideoCapture(src_video_path)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    # get fps and size of src video
    fps = vc.get(cv2.CAP_PROP_FPS)
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps:{} ".format(fps))
    print("size:{} ".format(size))
    # save picture after reid
    RESULT_FOLDER = dst_pic_path
    try:
        shutil.rmtree(RESULT_FOLDER)
    except OSError:
        pass
    os.mkdir(RESULT_FOLDER)
    # save video after reid
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(dst_video_path, fourcc, fps, size)
    # initialize database []
    db_vec = []
    # store frame info
    info_dict = {}

    # count frame
    frame_counter = 1
    # ----main process----
    set_gpu_ids(gpu_ids)
    pose_model = load_openpose()
    svd_model = load_reid()
    with torch.no_grad():
        while rval:
            initial_time = time.time()
            # return the relationship between the detected id
            # posebox is of the dim (N, width, length, 3)
            # coord is the box coord, where the id is drawn above

            # skeleton, posebox, joint_coords_db, coord = pose_si_processing(pose_model, frame, pose_val)
            skeleton, posebox, joint_coords_db, coord = pose_si_processing(pose_model, frame, pose_val, plot_skeleton)

            pose_time = time.time() - initial_time

            # determine if the current frame detects a person.
            # Yes: compare to the db_ved.
            # No: directly save picture.
            if len(posebox) != 0:
                id_vec = extract_feature(svd_model, posebox)
                people_num = len(id_vec)
                # require by json file
                info_dict_temp = {}
                video_info = defaultdict(list)
                info_dict_temp['num'] = str(frame_counter)
                info_dict_temp['result'] = video_info
                info_dict['frame_{}'.format(frame_counter)] = info_dict_temp

                # initialize the database if it's null
                if len(db_vec) == 0:
                    db_vec = id_vec
                    id_relation = np.arange(people_num)
                    print('Initialize the database on frame {}'.format(frame_counter), file = f)

                    # store info in json file
                    for i in range(len(id_vec)):
                        video_info['id'].append(str(float(id_relation[i])))
                        video_info['coords'].append(str(joint_coords_db[i]).replace('\n',''))

                    reid_time = time.time() - initial_time - pose_time
                    skeleton = draw_id(skeleton, coord, id_relation)
                    skeleton = putText(skeleton, people_num, frame_counter, pose_time, reid_time)


                else:
                    # jump into the match process
                    # id_relation: the relation between id and db
                    db_vec, id_relation = match(db_vec, id_vec, svd_val, f)
                    # store info in json file
                    for i in range(len(id_vec)):
                        video_info['id'].append(str(id_relation[i]))
                        video_info['coords'].append(str(joint_coords_db[i]).replace('\n',''))

                    print('***Current database length:{}***'.format(len(db_vec)), file=f)

                    reid_time = time.time() - initial_time - pose_time

                    skeleton = draw_id(skeleton, coord, id_relation)

                    skeleton = putText(skeleton, people_num, frame_counter, pose_time, reid_time)

                cv2.imwrite(RESULT_FOLDER + '/' + str("%04d" % frame_counter) + "skeleton.jpg", skeleton)

                videoWriter.write(skeleton)
                print('-----------------------------------------------', file=f)
                print("Frame " + str(frame_counter) + " completed in " + str(
                    round(time.time() - initial_time, 2)) + " s..." + '\n', file=f)
            else:
                people_num = 0
                reid_time = 0
                skeleton = putText(frame, people_num, frame_counter, pose_time, reid_time)
                cv2.imwrite(RESULT_FOLDER + '/' + str("%04d" % frame_counter) + "skeleton.jpg", skeleton)
                videoWriter.write(skeleton)
                print('-----------------------------------------------', file=f)
                print("Frame " + str(frame_counter) + " completed in " + str(
                    round(time.time() - initial_time, 2)) + " s..." + '\n', file=f)
            frame_counter += 1
            rval, frame = vc.read()
    # write in json file
    json_str = json.dumps(info_dict, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)

    videoWriter.release()
    f.close()
