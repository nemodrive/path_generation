from __future__ import division
import torch
import torch.nn.functional as F
import torchgeometry

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]



def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, rot, tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        rot: rotation matrix of cameras -- [B, 3, 4]
        tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if rot is not None:
        pcoords = rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if tr is not None:
        pcoords = pcoords + tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsic, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsic: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsic, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsic.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsic @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def main():
    from PIL import Image
    import numpy as np
    import cv2
    import h5py
    import matplotlib.pyplot as plt

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # img_file = '/HDD1_2TB/storage/dimlrgbd/train/HR/12. Livingroom/color/in_01_160217_123138_c.png'
    # img = Image.open(img_file)
    # img = np.array(img)
    # print(img.shape)
    # cv2.imshow('original', img)
    # cv2.waitKey(0)
    # img_tensor = torch.tensor(img).unsqueeze(0).float()
    # img_tensor = img_tensor.permute(0, 3, 1, 2) / 255.0
    #
    # depth_file = '/HDD1_2TB/storage/dimlrgbd/train/HR/12. Livingroom/depth_filled/in_01_160217_123138_depth_filled.png'
    # depth = Image.open(depth_file)
    # depth = np.array(depth)
    # depth_tensor = torch.tensor(depth).unsqueeze(0).float()
    # print(depth_tensor.max())
    # # depth_tensor /= depth_tensor.max()
    # # cv2.imshow('depth', depth)
    # # cv2.waitKey(0)
    #
    # pose = torch.tensor([[200.0, -200.0, -150.0, -0.0, -0.35, 0.35]]).float()
    #
    # intrinsic_file = '/HDD1_2TB/storage/rgbd/band_aid_clear_strips/calibration.h5'
    # h5_intrinsic = h5py.File(intrinsic_file)
    # # for a, b in h5_intrinsic.items():
    # #     print(a, b)
    #
    # intrinsic = torch.tensor(np.array([[1081.37, 1, 959.5], [0, 1081.37, 539.5], [0, 0, 1]])).unsqueeze(0).float()
    #
    # print(intrinsic)
    #
    # print(img_tensor.shape, depth_tensor.shape, intrinsic.shape)
    #
    # res, valid_points = inverse_warp(img_tensor, depth_tensor, pose, intrinsic)
    #
    # print(res.shape, valid_points.shape)
    #
    # res = res.cpu().numpy()[0]
    # res = np.transpose(res, (1, 2, 0))
    # cv2.imshow('res', res)
    # cv2.waitKey(0)

    # img_file = '/HDD1_2TB/storage/rgbd/band_aid_clear_strips/NP1_0.jpg'
    # img = Image.open(img_file)
    # img = np.array(img)
    # #img = cv2.resize(img, (640, 480))
    # cv2.imshow('original', img)
    # cv2.waitKey(0)
    # img_tensor = torch.tensor(img).unsqueeze(0).float()
    # img_tensor = img_tensor.permute(0, 3, 1, 2) / 255.0
    #
    # depth_file = '/HDD1_2TB/storage/rgbd/band_aid_clear_strips/NP1_0.h5'
    # h5_depth = h5py.File(depth_file)
    # depth_tensor = torch.tensor(np.array(h5_depth.get('depth')).astype(float)).unsqueeze(0).float()
    # # depth = np.array(depth).astype(float)
    # print(depth_tensor.max())
    # depth_tensor[depth_tensor == 0] = 1
    # depth_tensor /= depth_tensor.max()
    # # cv2.imshow('depth', depth)
    # # cv2.waitKey(0)
    #
    # pose = torch.tensor([[0.0, -0.0, -0.0, -0.0, 0.09, -0.0]]).float()
    #
    # intrinsic_file = '/HDD1_2TB/storage/rgbd/band_aid_clear_strips/calibration.h5'
    # h5_intrinsic = h5py.File(intrinsic_file)
    # # for a, b in h5_intrinsic.items():
    # #     print(a, b)
    #
    # intrinsic = torch.tensor(np.array(h5_intrinsic.get('N1_rgb_K')).astype(float)).unsqueeze(0).float() / 2.0
    #
    # print(intrinsic)
    #
    # print(img_tensor.shape, depth_tensor.shape, intrinsic.shape)
    #
    # res, valid_points = inverse_warp(img_tensor, depth_tensor, pose, intrinsic)
    #
    # print(res.shape, valid_points.shape)

    # res = res.cpu().numpy()[0]
    # res = np.transpose(res, (1, 2, 0))
    # cv2.imshow('res', res)
    # cv2.waitKey(0)

    # draw cube
    w = 512
    hw = w // 2
    depth = np.zeros((w, w), dtype=np.uint8)
    img = np.zeros((w, w, 3), dtype=float)
    depth.fill(10)

    cube_front_w = 100 // 2
    cube_back_w = 40 // 2
    cube_depth_w = 20
    cube_start_depth = 1
    cube_back_depth = 7

    depth[hw - cube_front_w: hw + cube_front_w, hw - cube_front_w: hw + cube_front_w] = cube_start_depth

    depths = np.linspace(cube_back_depth, cube_start_depth, cube_depth_w)
    depth_ws = np.linspace(cube_back_w, cube_front_w, cube_depth_w)
    for i, row in enumerate(range(hw - cube_front_w - cube_depth_w, hw - cube_front_w)):
        offset = int(depth_ws[i])
        depth[row, hw - offset: hw + offset] = depths[i]

    depth = depth * (255 // depth.max())

    cv2.imshow("depth", depth)
    cv2.waitKey(0)

    for i in range(w):
        for j in range(w):
            img[i, j, 2] = 255 - depth[i, j]
            if img[i, j, 2] < 10:
                img[i, j, :] = 255

    img /= 255.0

    cv2.imshow("rgb", img)
    cv2.waitKey(0)

    img_tensor = torch.tensor(img).unsqueeze(0).float()
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    depth_tensor = torch.tensor(depth).unsqueeze(0).float()

    pose = torch.tensor([[-0.1, 0.3, -0.0, 0., -0., 0.5]]).float()

    intrinsic = torch.tensor(np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
                                       [0.000000e+00, 9.569251e+02, 2.241806e+02],
                                       [0.000000e+00, 0.000000e+00, 1.000000e+00]])).unsqueeze(0).float()

    def nothing(x):
        pass

    cv2.namedWindow('image')

    ticks = 600
    middle = ticks // 2

    # create trackbars for color change
    cv2.createTrackbar('tx', 'image', middle, ticks, nothing)
    cv2.createTrackbar('ty', 'image', middle, ticks, nothing)
    cv2.createTrackbar('tz', 'image', middle, ticks, nothing)
    cv2.createTrackbar('rx', 'image', middle, ticks, nothing)
    cv2.createTrackbar('ry', 'image', middle, ticks, nothing)
    cv2.createTrackbar('rz', 'image', middle, ticks, nothing)

    while True:
        res, valid_points = inverse_warp(img_tensor, depth_tensor, pose, intrinsic)
        res = res.cpu().numpy()[0]
        res = np.transpose(res, (1, 2, 0))
        cv2.imshow('image', res)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        tx = (cv2.getTrackbarPos('tx', 'image') - middle) / 100
        ty = (cv2.getTrackbarPos('ty', 'image') - middle) / 100
        tz = (cv2.getTrackbarPos('tz', 'image') - middle) / 100
        rx = (cv2.getTrackbarPos('rx', 'image') - middle) / 100
        ry = (cv2.getTrackbarPos('ry', 'image') - middle) / 100
        rz = (cv2.getTrackbarPos('rz', 'image') - middle) / 100

        pose = torch.tensor([[tx, ty, tz, rx, ry, rz]]).float()


if __name__ == '__main__':
    main()
