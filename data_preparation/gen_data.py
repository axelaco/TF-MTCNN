import cv2 as cv
import sys
import numpy.random as npr
import math
import numpy as np
import tqdm
from prepare_utils import multiple_IOU
import os

min_size_percentage = 0.7
max_size_percentage = 1.1
offset_center_percentage = 0.2

WIDER_FACE_FOLDER = '/home/axel/Documents/MTCNN/WIDER_train/images'
DATA_GENERATED_FOLDER = '/home/axel/Documents/MTCNN/data/'



def parse_annotation_file(ann_file, pos_img_folder, part_img_folder, neg_img_folder):
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    pos_idx = 0
    part_idx = 0
    neg_idx = 0

    pbar = tqdm.tqdm(total=int(len(lines)))

    while(True):    
        if idx >= len(lines):
            break
        img_file = os.path.join(WIDER_FACE_FOLDER, lines[idx].strip())
        idx += 1
        bbox_gt = int(lines[idx])
        if bbox_gt > 0:
            bboxes = []
            idx += 1

            for i in range(bbox_gt):
                line = lines[idx].strip().split(' ')[:4]
                line = list(map(lambda x: int(x), line))
                bboxes.append(np.array(line, dtype=np.int32))
                idx += 1
            bboxes = np.array(bboxes)
            img = cv.imread(img_file)
            f = open(os.path.join(DATA_GENERATED_FOLDER, str(net_size)) + '/' + 'annotation_file.txt', 'a+')
            neg_idx = generate_neg_image(img=img, gt_bboxes=bboxes, neg_img_folder=neg_img_folder, neg_idx=neg_idx, net_size=24, f=f)
            pos_idx, part_idx = generate_bbox_image(img=img, gt_bboxes=bboxes, pos_img_folder=pos_img_folder, part_img_folder=part_img_folder, pos_idx=pos_idx, part_idx=part_idx, f=f)
            pbar.update(int(bbox_gt))
        else:
            pbar.update(2)
            idx += 2
    print("pos={}, part={}, neg={}".format(pos_idx, part_idx, neg_idx))

def generate_bbox_image(img, gt_bboxes, pos_img_folder, part_img_folder, pos_idx, part_idx, net_size=12, num_pos=20, f=None):
    width, height, channels = img.shape
    for box in gt_bboxes:
        x, y, w, h = box
        
        if max(w, h) < 40 or x < 0 or y < 0:
                continue
        
        box_center_x = x + w / 2
        box_center_y = y + h / 2

        for i in range(num_pos):
            new_bbox_size = npr.randint(min(w, h) * min_size_percentage, max(w, h) * max_size_percentage)

            w_new = w * offset_center_percentage
            h_new = h * offset_center_percentage

            if (box_center_x - w_new >= box_center_x + w_new) or (box_center_y - h_new >= box_center_y + h_new):
                continue

            new_center_x = npr.randint(box_center_x - w_new, box_center_x + w_new)

            new_center_y = npr.randint(box_center_y - h_new , box_center_y + h_new)

            nx1 = int(new_center_x - new_bbox_size / 2)
            ny1 = int(new_center_y - new_bbox_size / 2)

            nx2 = nx1 + new_bbox_size
            ny2 = ny1 + new_bbox_size

            if nx2 > width or ny2 > height or nx2 < 0 or ny2 < 0:
                    continue

            crop_box = np.array([nx1, ny1, new_bbox_size, new_bbox_size])


            offset_x1 = (x - nx1) / float(new_bbox_size)
            offset_x2 = (x + w - nx2) / float(new_bbox_size)
            offset_y1 = (y - ny1) / float(new_bbox_size)
            offset_y2 = (y + h - ny2) / float(new_bbox_size)

            cropped_img = img[ny1:ny2, nx1:nx2, :]

            if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                continue

            
            resized_img = cv.resize(cropped_img, (net_size, net_size), interpolation=cv.INTER_LINEAR)
            
            if np.max(multiple_IOU(crop_box, gt_bboxes)) >= 0.65:
                img_path = pos_img_folder + '/{}.jpg'.format(pos_idx)
                cv.imwrite(img_path, resized_img)
                
                ann_str = '{img_path} {label} {x1} {y1} {x2} {y2}\n' 
                f.write(ann_str.format(img_path=img_path, label=1, x1=offset_x1, y1=offset_y1, x2=offset_x2, y2=offset_y2))

                pos_idx += 1
            elif np.max(multiple_IOU(crop_box, gt_bboxes)) >= 0.4:
                img_path = part_img_folder + '/{}.jpg'.format(part_idx)
                cv.imwrite(img_path, resized_img)
                
                ann_str = '{img_path} {label} {x1} {y1} {x2} {y2}\n' 
                f.write(ann_str.format(img_path=img_path, label=-1, x1=offset_x1, y1=offset_y1, x2=offset_x2, y2=offset_y2))

                
                part_idx += 1
    return pos_idx, part_idx

def generate_neg_image(img, gt_bboxes, neg_img_folder, neg_idx, net_size=12, num_neg=40, f=None):
    width, height, channels = img.shape
    num = 0
    while num < num_neg:
        new_bbox_size = npr.randint(12, 40)
        nx1 = npr.randint(0, width - new_bbox_size)  # Note: randint() is from uniform distribution.
        ny1 = npr.randint(0, height - new_bbox_size)

        nx2 = nx1 + new_bbox_size
        ny2 = ny1 + new_bbox_size
        
        if nx2 > width or ny2 > height:
            continue
        
        crop_box = np.array([nx1, ny1, new_bbox_size, new_bbox_size])

        cropped_img = img[ny1:ny2, nx1:nx2, :]

        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            continue

        resized_img = cv.resize(cropped_img, (net_size, net_size), interpolation=cv.INTER_LINEAR)    


        if np.max(multiple_IOU(crop_box, gt_bboxes)) <= 0.3:
            img_path = neg_img_folder + '/{}.jpg'.format(neg_idx)
            cv.imwrite(img_path, cropped_img)    
            ann_str = '{img_path} {label}\n' 
            f.write(ann_str.format(img_path=img_path, label=0))

            num += 1
            neg_idx += 1
        
    return neg_idx


if __name__ == '__main__':
    args = sys.argv[1:]

    net_size = 24
    if not os.path.exists(DATA_GENERATED_FOLDER):
        os.makedirs(DATA_GENERATED_FOLDER)


    pos_img_folder = os.path.join(DATA_GENERATED_FOLDER, str(net_size), 'img_pos')
    if not os.path.exists(pos_img_folder):
        os.makedirs(pos_img_folder)

    part_img_folder = os.path.join(DATA_GENERATED_FOLDER, str(net_size), 'img_part')
    if not os.path.exists(part_img_folder):
        os.makedirs(part_img_folder)

    neg_img_folder = os.path.join(DATA_GENERATED_FOLDER, str(net_size), 'img_neg')
    if not os.path.exists(neg_img_folder):
        os.makedirs(neg_img_folder)

    parse_annotation_file(args[0], pos_img_folder, part_img_folder, neg_img_folder)
    """
    for key, value in tqdm.tqdm(ann_res.items()):
        img = cv.imread(key)
        neg_idx = generate_neg_image(img, value, neg_img_folder, neg_idx)
        pos_idx, part_idx = generate_bbox_image(img, value, pos_img_folder, part_img_folder, pos_idx=pos_idx, part_idx=part_idx)
    print("pos={}, part={}, neg={}".format(pos_idx, part_idx, neg_idx))

    
    
    min_size_percentage = 0.8
    max_size_percentage = 1.2
    offset_center_percentage = 0.3
    img = cv.imread(args[0])

    
    gt = [449,330,122,149]

    img = cv.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 0, 0), 1)

    for i in range(1):
        w, h = gt[2], gt[3]

        box_center_x = gt[0] + w / 2

        box_center_y = gt[1] + h / 2

        size = npr.randint(min(w, h) * min_size_percentage, max(w, h) * max_size_percentage)
    
        w_new = w * offset_center_percentage
        h_new = h * offset_center_percentage

        new_center_x = npr.randint(box_center_x - w_new, box_center_x + w_new)

        new_center_y = npr.randint(box_center_y - h_new , box_center_y + h_new)

        nx1 = int(new_center_x - size / 2)
        ny1 = int(new_center_y - size / 2)

        nx2 = nx1 + size
        ny2 = ny1 + size 

        
        img = cv.rectangle(img, (nx1, ny1), (nx2, ny2), (0, 0, 255), 3)
    
        offset_x1 = (gt[0] - nx1) / float(size)
        offset_y1 = (gt[1] - ny1) / float(size)
        offset_x2 = (gt[0] + gt[2] - nx2) / float(size)
        offset_y2 = (gt[1] + gt[3] - ny2) / float(size)
        
        
        print(offset_x1, offset_x2, offset_y1, offset_yt2)
        cropped_img = img[new_y:new_y + size, new_x:new_x + size, :]
  

        cv.imshow('cropped', cropped_img)
        
        cv.waitKey(0)

       
    cv.imshow('frame', img)
   
    cv.waitKey(0)
    """
