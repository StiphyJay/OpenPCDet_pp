"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import matplotlib
import open3d
import numpy as np
from . import kitti_util as utils
import pdb

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def text_3d(text, pos, direction=None, degree=0.0, font='/mnt/c/Windows/Fonts/YuGothB.ttc', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (1., 0., 0.)
    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = open3d.geometry.PointCloud()
    pcd.colors = open3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = open3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

def draw_scenes(points, gt_boxes=None, pred_boxes=None, calib=None, base_boxes=None, pred_labels=None, pred_scores=None, point_colors=None, draw_origin=True):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3) #np.zeros(3) #北京颜色为白

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        # pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3))) #点云颜色为黑
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1), calib, is_gt=True)

    if pred_boxes is not None:
        vis = draw_box(vis, pred_boxes, (0, 1, 0), calib, pred_labels, pred_scores)

    if base_boxes is not None:
        vis = draw_box(vis, base_boxes, (1, 0, 0), calib, pred_labels, pred_scores)


    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes, calib):
    center = np.array(gt_boxes.t)
    lwh = np.array([gt_boxes.l, gt_boxes.h, gt_boxes.w])
    axis_angles = np.array([0, 0, gt_boxes.ry + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    #from visualize.py
    _, box3d_pts_3d = utils.compute_box_3d(gt_boxes, calib.P)
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(box3d_pts_3d_velo)
    # import ipdb; ipdb.set_trace(context=20)
    #define corners link
    """
            4 -------- 5
           /|         /|
          7 -------- 6 .
          | |        | |
          . 0 -------- 1
          |/         |/
          3 -------- 2
    """
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[0, 1], [1, 2], [2, 3], [3, 0],[4, 5], [5, 6], [6, 7], [4, 7], [0, 4],[3, 7], [1, 5], [2, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), calib=None, ref_labels=None, score=None, is_gt=False):
    for obj in gt_boxes:
        if obj.type == "DontCare":
            continue
        line_set, box3d = translate_boxes_to_open3d_instance(obj, calib)
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        if not is_gt:
            score = obj.scores
            vis.add_geometry(text_3d(text=str(score), pos=np.asarray(line_set.points)[4]))
    return vis
