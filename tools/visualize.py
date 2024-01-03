import argparse
import glob
from pathlib import Path
import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
from visual_utils import kitti_util as utils
OPEN3D_FLAG = True
import pickle
import numpy as np
import torch
import os
import pdb
    
class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, pred_path, args=None):
        """root_dir contains training and testing folders"""
        self.label_dir = '/home/zhousifan/OpenPCDet/data/kitti/training/label_2/'
        self.lidar_dir = '/home/zhousifan/OpenPCDet/data/kitti/training/velodyne/'
        self.calib_dir = '/home/zhousifan/OpenPCDet/data/kitti/training/calib'
        self.pred_dir = pred_path
        self.num_samples = len(os.listdir(self.pred_dir))

    def __len__(self):
        return self.num_samples

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_pred_objects(self, idx):
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None


def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    objects_pred=None,
):
#show_lidar_with_boxes(pc_velo, objects, calib, objects_pred)
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(600, 600)
    )
    
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    # mlab.show(1)
    mlab.show(stop=True)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    args = parser.parse_args()

    return args

def main():
    args = parse_config()
    pred_path = '/home/zhousifan/OpenPCDet/output/home/zhousifan/OpenPCDet/tools/cfgs/kitti_models/HA_pointpillar_64_128o_fixs/default/eval/epoch_80/val/default/final_result/data'
    dataset = kitti_object(pred_path, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset
        data_idx = 8 #scene id
        objects = dataset.get_label_objects(data_idx)
        objects_pred = dataset.get_pred_objects(data_idx)
        n_vec = 4
        dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1
        show_lidar_with_boxes(pc_velo, objects, calib, objects_pred)

if __name__ == '__main__':
    main()
