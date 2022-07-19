#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image
import imageio
# import OpenEXR
import struct
import os

## print top 10 values of mse in blue color
def draw_top_10_values_in_image(testsavedir,mse_map_show,rgb_pred,unc_map_show,img_i):

    mse_max = np.amax(mse_map_show)
    mse_ind_h, mse_ind_w = np.where(mse_map_show == mse_max)
    rgb_pred_draw_max = cv2.circle(rgb_pred, (mse_ind_w[0], mse_ind_h[0]), radius=3, color=(255, 0, 0), thickness=1)
    var_max = np.amax(unc_map_show)
    var_ind_h, var_ind_w = np.where(unc_map_show == var_max)
    # rgb_pred_draw_max = cv2.circle(rgb_pred_draw_max, (var_ind_w[0],var_ind_h[0]), radius=3, color=(0, 0, 255), thickness=1)
    for m in range(10): 
        print('mse max:',mse_max)
        print('mse max ind:', (mse_ind_w,mse_ind_h))
        print('var max:',var_max)
        print('var max ind:', (var_ind_w,var_ind_h))

        mse_map_show[mse_ind_h, mse_ind_w] -= 10
        unc_map_show[var_ind_h, var_ind_w] -= 10

        mse_max = np.amax(mse_map_show)
        mse_ind_h, mse_ind_w = np.where(mse_map_show == mse_max)
        var_max = np.amax(unc_map_show)
        var_ind_h, var_ind_w = np.where(unc_map_show == var_max)

        rgb_pred_draw_max = cv2.circle(rgb_pred_draw_max, (mse_ind_w[0],mse_ind_h[0]), radius=3, color=(255, 0, 0), thickness=1)
        # rgb_pred_draw_max = cv2.circle(rgb_pred_draw_max, (var_ind_w[0],var_ind_h[0]), radius=3, color=(0, 0, 255), thickness=1)

    cv2.imwrite(testsavedir+'/{:02d}_pred_draw_max.png'.format(img_i), rgb_pred_draw_max)


def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()