import argparse
from pathlib import Path
# import PyQt5
import open3d
from visual_utils import open3d_vis_utilsV2 as V
OPEN3D_FLAG = True
from visual_utils import kitti_util as utils
import numpy as np
import os
import pdb
    
class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, pred_path, args=None):
        """root_dir contains training and testing folders"""
        self.label_dir = '/root/OpenPCDet_pp/data/kitti/training/label_2'
        self.lidar_dir = '/root/OpenPCDet_pp/data/kitti/training/velodyne/'
        self.calib_dir = '/root/OpenPCDet_pp/data/kitti/training/calib/'
        self.pred_dir = pred_path
        self.baseline_dir = '/root/OpenPCDet_pp/output/pointpillars/pred_data/'
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

    def get_base_objects(self, idx):
        baseline_filename = os.path.join(self.baseline_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(baseline_filename)
        if is_exist:
            return utils.read_label(baseline_filename)
        else:
            return None


def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    objects_pred=None,
    objects_base=None,
):
#show_lidar_with_boxes(pc_velo, objects, calib, objects_pred)
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    print(("All point num: ", pc_velo.shape[0]))
    # print("pc_velo", pc_velo.shape)
    V.draw_scenes(points=pc_velo, gt_boxes=objects, pred_boxes=objects_pred, calib=calib, base_boxes=objects_base)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    args = parser.parse_args()

    return args

def main():
    args = parse_config()
    pred_path = '/root/OpenPCDet_pp/output/HA_pointpillars_64_128o_fixs/pred_data/'
    dataset = kitti_object(pred_path, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset
        data_idx = 40 #scene id
        objects = dataset.get_label_objects(data_idx)
        objects_pred = dataset.get_pred_objects(data_idx)
        objects_base = dataset.get_base_objects(data_idx)
        n_vec = 4
        dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                # print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1
        show_lidar_with_boxes(pc_velo, objects, calib, objects_pred, objects_base)

if __name__ == '__main__':
    main()
