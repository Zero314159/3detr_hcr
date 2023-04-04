import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points

import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)
from .annotation import Annotation3D, Annotation3DPly, global2local
from .ply        import read_ply
import math

DATA_PATH = "/home/hcr/Data/KITTI-360/" ## Replace with path to dataset

def loadPoses(pos_file):
  ''' load system poses '''

  data = np.loadtxt(pos_file)
  ts = data[:, 0].astype(np.int)
  poses = np.reshape(data[:, 1:], (-1, 3, 4))
  poses = np.concatenate((poses, np.tile(np.array([0, 0, 0, 1]).reshape(1,1,4),(poses.shape[0],1,1))), 1)
  return ts, poses

kitti2me = {
    12 : 0,#'car'
    13 : 1,#'truck'
    14 : 2,#'trailer'
    16 : 3,#'motorcycle'
    17 : 4,#'bicycle'
    18 : 5,#'person'
}

class Kitti360DatasetConfig(object):
    def __init__(self):
        self.num_semcls = 6
        self.num_angle_bin = 2
        self.max_num_obj = 20
        self.type2class = {
            'car': 0,
            'truck': 1,
            'trailer': 2,
            'motorcycle': 3,
            'bicycle': 4,
            'person': 5,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        '''convert (center, size, angle) to 8 corners of box.'''
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

class Kitti360DetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        meta_data_dir=None,
        num_points=50000,
        use_color=False,
        use_height=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=50000,
    ):
        assert num_points <= 250000
        assert split_set in ["train", "val"]
        self.dataset_config = dataset_config
        
        if root_dir is None:
            root_dir = DATA_PATH

        self.seqs = [0, 2, 3, 4, 5, 6, 7, 9, 10]

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 20

        self.point_cloud_dataset = []
        self.bbox_dataset = []

        # obj_cnt = {}
        # minl, minw, minh, lowest_top = 999999.,999999.,999999., 999999.
        # maxl, maxw, maxh, highest_bottom = -999999.,-999999.,-999999., -999999.
        # top_list = []
        # bottom_list = []
        # cls_cnt = np.zeros(11)

        for seq in self.seqs:
            sequence = '2013_05_28_drive_%04d_sync' % seq
            label3DPcdPath  = os.path.join(root_dir, 'data_3d_semantics')
            label3DBboxPath = os.path.join(root_dir, 'data_3d_bboxes')
            annotation3D = Annotation3D(label3DBboxPath, sequence)
            annotation3DPly = Annotation3DPly(label3DPcdPath, sequence)
            posePath = os.path.join(root_dir, 'data_poses', sequence, 'poses.txt')

            ts, poses = loadPoses(posePath)
            ts_idx = 0

            all_bboxes = []
            semanticlabels = []
            bboxes_window = []
            for globalId,v in annotation3D.objects.items():
                # skip dynamic objects
                if len(v)>1:
                    continue
                for obj in v.values():
                    all_bboxes.append(obj.vertices)
                    semanticId, instanceId = global2local(globalId)
                    semanticlabels.append(semanticId)
                    bboxes_window.append([obj.start_frame, obj.end_frame])
            
            if split_set == 'train':
                pcdFilelist = annotation3DPly.pcdFileList[:int(0.7 * len(annotation3DPly.pcdFileList))]
                # pcdFilelist = annotation3DPly.pcdFileList
            else:
                pcdFilelist = annotation3DPly.pcdFileList[int(0.7 * len(annotation3DPly.pcdFileList)):]
            
            for pcdFile in pcdFilelist:
                window = pcdFile.split(os.sep)[-1]
                window = window.split('_')
                window[1] = window[1].split('.')[0]
                window = [int(i) for i in window]

                data = read_ply(pcdFile)
                pcd_np = np.vstack((data['x'], data['y'], data['z'])).T

                while ts[ts_idx] < window[0]:
                    ts_idx += 1
                while ts[ts_idx] <= window[1]:
                # x_cur = np.min(pcd_np, axis=0)[0]
                # while x_cur + 20. < np.max(pcd_np, axis=0)[0]:
                #     y_cur = np.min(pcd_np, axis=0)[1]
                #     while y_cur + 20. < np.max(pcd_np, axis=0)[1]:
                        pcd_center = poses[ts_idx][:3, -1]
                        ts_idx += 10

                        pcd_np_part = pcd_np[pcd_np[:, 0] > pcd_center[0] - 10.]
                        pcd_np_part = pcd_np_part[pcd_np_part[:, 0] < pcd_center[0] + 10.]
                        pcd_np_part = pcd_np_part[pcd_np_part[:, 1] > pcd_center[1] - 10.]
                        pcd_np_part = pcd_np_part[pcd_np_part[:, 1] < pcd_center[1] + 10.]

                        if pcd_np_part.shape[0] > 50000:# and pcd_np_part.shape[0] < 250000:

                            # pcd_center = np.array([x_cur + 10., y_cur + 10., np.min(pcd_np_part, axis=0)[2]])
                            pcd_np_part = pcd_np_part - pcd_center
                            
                            bbox_part = []
                            for i in range(len(all_bboxes)):
                                if bboxes_window[i][0] == window[0]:
                                    corners = all_bboxes[i]
                                    if np.average(corners, axis=0)[0] > pcd_center[0] - 10. and np.average(corners, axis=0)[1] > pcd_center[1] - 10. and \
                                        np.average(corners, axis=0)[0] < pcd_center[0] + 10. and np.average(corners, axis=0)[1] < pcd_center[1] + 10. and \
                                        semanticlabels[i] in kitti2me.keys():

                                        # print(corners)
                                        corners = corners - pcd_center
                                        # print(pcd_center)
                                        # print(corners)
                                        center = np.average(corners, axis=0)
                                        l = np.linalg.norm(corners[0, :] - corners[5, :])
                                        w = np.linalg.norm(corners[0, :] - corners[2, :])
                                        h = np.linalg.norm(corners[0, :] - corners[1, :])
                                        heading_vector = corners[0, :] - corners[5, :]
                                        heading_angle = math.atan2(heading_vector[1], heading_vector[0] + 1e-10)
                                        bbox = [center[0], center[1], center[2], 
                                                l/2., w/2., h/2., 
                                                -heading_angle, kitti2me[semanticlabels[i]]]
                                        bbox_part.append(bbox)

                                        # cls_cnt[kitti2me[semanticlabels[i]]] += 1
                                        # minl = min(minl, l)
                                        # minw = min(minw, w)
                                        # minh = min(minh, h)
                                        # maxl = max(maxl, l)
                                        # maxw = max(maxw, w)
                                        # maxh = max(maxh, h)
                                        # bottom = np.min(corners, axis=0)[2]
                                        # top = np.max(corners, axis=0)[2]
                                        # # highest_bottom = max(highest_bottom, bottom)
                                        # # lowest_top = min(lowest_top, top)
                                        # top_list.append(top)
                                        # bottom_list.append(bottom)
                                        # # if bottom > 5.:
                                        # #     print(pcd_center)
                                        # #     print(corners)
                                        # #     print(semanticlabels[i])
                                        
                            if len(bbox_part) > 0:
                                bbox_part = np.array(bbox_part)
                                self.point_cloud_dataset.append(pcd_np_part)
                                self.bbox_dataset.append(bbox_part)

                                # if not len(bbox_part) in obj_cnt.keys():
                                #     obj_cnt[len(bbox_part)] = 1
                                # else:
                                #     obj_cnt[len(bbox_part)] += 1

                    #     y_cur += 20.
                    # x_cur += 20.
        
        # print(obj_cnt)
        # print("l:   ", minl, maxl)
        # print("w:   ", minw, maxw)
        # print("h:   ", minh, maxh)
        # # print("highest bottom:   ", highest_bottom)
        # # print("lowest top:   ", lowest_top)
        # print()
        # print(sorted(bottom_list)[-100:])
        # print('\n\n\n')
        # print(sorted(top_list, reverse=True)[-100:])

        # for i in range(len(cls_cnt)):
        #     print(dataset_config.class2type[i], ": ", cls_cnt[i])
        # exit(0)

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, idx):
        point_cloud = self.point_cloud_dataset[idx]
        bboxes = self.bbox_dataset[idx]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                bboxes[:, 1] = -1 * bboxes[:, 1]
                bboxes[:, 6] = -bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment point cloud scale: 0.85x-1.15x
            # scale_ratio = np.random.random() * 0.3 + 0.85
            # scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            # point_cloud[:, 0:3] *= scale_ratio
            # bboxes[:, 0:3] *= scale_ratio
            # bboxes[:, 3:6] *= scale_ratio

            # if self.use_height:
            #     point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:
                point_cloud, bboxes, _ = self.random_cuboid_augmentor(
                    point_cloud, bboxes
                )
        
        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : bboxes.shape[0]] = 1
        max_bboxes = np.zeros((self.max_num_obj, 8))
        max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_size = bbox[3:6] * 2.0
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox
        
        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 7
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        return ret_dict