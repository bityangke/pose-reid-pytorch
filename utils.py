# coding=utf-8
# Created by utils on 2019-03-11 21:41
# Copyright © 2019 Alan. All rights reserved.
import numpy as np
import cv2
import math


def match(db_vec, id_vec, val, f):
    db_len = len(db_vec)
    id_len = len(id_vec)
    id_relation = np.zeros(id_len)
    id_relation[:] = -1
    # record the currently matched value
    id_mapping = []
    # list for recording value less than val
    distance_less_than_val = []
    # every vec in id_vec
    for j in range(id_len):
        # every vec in db_vc
        for i in range(db_len):
            current_distance = np.sum(np.abs(db_vec[i] - id_vec[j]))
            distance_less_than_val.append((current_distance, i, j))
            # record smallest compare with db_vec
    distance_less_than_val.sort(reverse=True)
    distance_less_than_val_number = len(distance_less_than_val)
    # to judge that a picture cannot be two identical people
    for i in range(distance_less_than_val_number):
        distance, dest, id_vec_position = distance_less_than_val[distance_less_than_val_number - 1 - i]
        if distance < val:
            if dest not in id_relation and id_vec_position not in id_mapping:
                # rectify the final result
                id_mapping.append(id_vec_position)
                db_vec[dest] = (db_vec[dest] + id_vec[id_vec_position, :])/2
                id_relation[id_vec_position] = dest
                print('Detect people:({}|{})\tSamllest_distance:{}\tPeople_id:{}'.format(
                    j + 1, id_len, distance, dest), file=f)
                if len(id_mapping) == id_len:
                    break
            else:
                continue
        else:
            if id_vec_position in id_mapping:
                continue
            else:
                id_mapping.append(id_vec_position)
                db_vec = np.append(db_vec, id_vec[np.newaxis, id_vec_position, :], axis=0)
                id_relation[id_vec_position] = db_vec.shape[0] - 1
                print('Detect people(NEW):({}|{})\tSamllest_distance:{}\tPeople_id:{}'.format(j + 1, id_len, distance,
                                                                                              db_vec.shape[0] - 1),
                      file=f)
                if len(id_mapping) == id_len:
                    break
    if len(id_mapping) != id_len:
        for i in range(id_len):
            if i not in id_mapping:
                db_vec = np.append(db_vec, id_vec[np.newaxis, i, :], axis=0)
                id_relation[i] = db_vec.shape[0] - 1
                print('Detect people(NEW):({}|{})\tSmallest_distance:{}\tPeople_id:{}'.format(
                    j + 1, id_len, distance, db_vec.shape[0] - 1),file=f)
    return db_vec, id_relation

def draw_id(skeleton, coords, id_relation):
    scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    index = 0
    for coord in coords:
        color = (227 + 200 * math.sin(id_relation[index]), \
                 227 + 200 * math.sin(3 * id_relation[index] - 2), \
                 227 + 200 * math.cos(2 * id_relation[index]))
        # draw ids
        text_y = (coord[0] + coord[1]) // 2
        text_x = (coord[2] + coord[3]) // 2
        content = "id-" + str(int(id_relation[index]))
        cv2.putText(skeleton, content, (text_x, text_y), font, scale, color, 2)
        half_h = (coord[1] - coord[0])//2
        half_w = (coord[3] - coord[2])//2
        xmin = text_x - half_w
        ymin = text_y - half_h
        xmax = text_x + half_w
        ymax = text_y + half_h
        # draw rectangles
        skeleton = cv2.rectangle(skeleton, (xmin, ymin), (xmax, ymax), color)
        index += 1

    return skeleton


def putText(skeleton, people_num, frame_counter, pose_time, reid_time):

    scale = 0.5
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.LINE_AA
    content = "People number: " + str(people_num)
    cv2.putText(skeleton, content, (10, 10), font, scale, color)

    content = "Frame: " + str(frame_counter)
    cv2.putText(skeleton, content, (10, 30), font, scale, color)

    content = "Pose Time: " + str(round(pose_time, 2)) + 's'
    cv2.putText(skeleton, content, (10, 50), font, scale, color)

    content = "Reid Time: " + str(round(reid_time, 2)) + 's'
    cv2.putText(skeleton, content, (10, 70), font, scale, color)

    return skeleton
