import torch
import torch.nn.functional as F
from utils.warp_utils import euler2mat, quat2mat


def sample_pix_coord(img_width: int, img_height: int, device="cpu", batch: int = 0):
    r1 = torch.arange(0, img_height, device=device)
    r2 = torch.arange(0, img_width, device=device)
    ones = torch.ones(img_height, img_width, device=device).long()
    p = torch.stack(torch.meshgrid(r1, r2)[::-1] + (ones,), dim=2).view(-1, 3).float().t()

    if batch > 0:
        p = p.unsqueeze(0).expand(batch, *p.size())  # Expand to batch size

    return p


def transform_pose(pose: torch.Tensor, rotation_mode = 'euler'):
    """
    :param pose: dim [B, 6]
    :return: rot_and_translation
    """
    # ==============================================================================================
    # @ from SfmLearner 
    transp = pose[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = pose[:,3:]
    if rotation_mode == 'euler':
        rot = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot = quat2mat(rot)  # [B, 3, 3]

    # ==============================================================================================

    return rot, transp


def img_coord_2_homogenous(p_coords, normalize: bool = True, width: int = None, height: int = None,
                           clamp_z: float = 1e-3):
    z = p_coords[:, 2]
    if clamp_z > 0:
        z = z.clamp(min=1e-3)  # TODO why?

    img_coord = p_coords[:, :2] / z.unsqueeze(1)

    if normalize:
        x = img_coord[:, 0] / ((width - 1) / 2.) - 1.
        y = img_coord[:, 1] / ((height - 1) / 2.) - 1.
        img_coord = torch.stack([x, y], dim=2)

    return img_coord


def reverse_warp(img: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor,
                 intrinsic: torch.Tensor, distort_coeff: torch.Tensor,
                 rotation_type: str = "euler", padding_mode: str = "zeros"):
    """

    :param img:
    :param depth:
    :param pose:
    :param intrinsic:
    :param rotation_type:
    :param padding_mode:
    :return:
    """
    batch_size, channel, img_height, img_width = img.size()

    # Generate pixel coord  - homogeneous coordinates
    p = sample_pix_coord(img_width, img_height, device=img.device, batch=batch_size)

    # TODO could be useful to correct image for distortion

    rot, trans = transform_pose(pose)

    depth_v = depth.view(batch_size, 1, -1)

    homo_coord = intrinsic @ rot @ intrinsic.inverse() @ p * depth_v + intrinsic @ trans

    # Converting from homogeneous to image coordinates
    img_p = img_coord_2_homogenous(homo_coord, normalize=True, width=img_width, height=img_height)

    # Reshape 2 grid
    grid = img_p.reshape(batch_size, img_height, img_width, 2)

    # Get projected image
    projected_img = F.grid_sample(img, grid, padding_mode=padding_mode)
    valid_points = (grid.abs().max(dim=-1)[0] <= 1).unsqueeze(1)

    return projected_img, valid_points


def test_reverse_warp():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    import cv2

    global timg, depth, pose, intrinsic, sx, sy, sz, rx, ry, rz

    h, w = 100, 100
    f =  4.
    depth = torch.ones(1, h, w)
    depth[0, 40:60, 40:60] = 0.5

    intrinsic = torch.tensor(
        [[w /f , 0., w/2.],
        [0., w/f, h/2.],
        [0., 0., 1.]]
    ).unsqueeze(0)
    pose = torch.tensor([0, 0, 0, 0, 0, 0]).float().unsqueeze(0)

    img = cv2.imread("/home/nemodrive0/workspace/andrein/path_generation/dataset/0001.png")
    img = cv2.resize(img, (h, w))
    timg = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0)

    def test_warp(timg, depth, pose, intrinsic):
        new_img, _ = reverse_warp(timg, depth, pose, intrinsic)
        new_img = new_img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        return new_img

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.40)

    l = plt.imshow(img)

    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

    ax_rx = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_ry = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    ax_rz = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)

    f0 = 0.
    delta_f = 0.1
    interval = [-5, 5]
    sx = Slider(ax_x, 'X', interval[0], interval[1], valinit=f0, valstep=delta_f)
    sy = Slider(ax_y, 'Y', interval[0], interval[1], valinit=f0, valstep=delta_f)
    sz = Slider(ax_z, 'Z', interval[0], interval[1], valinit=f0, valstep=delta_f)
    rx = Slider(ax_rx, 'Rx', interval[0], interval[1], valinit=f0, valstep=delta_f)
    ry = Slider(ax_ry, 'Ry', interval[0], interval[1], valinit=f0, valstep=delta_f)
    rz = Slider(ax_rz, 'Rz', interval[0], interval[1], valinit=f0, valstep=delta_f)

    def update(val):
        global timg, depth, pose, intrinsic, sx, sy, sz, rx, ry, rz
        x = sx.val
        y = sy.val
        z = sz.val
        vrx = rx.val
        vry = ry.val
        vrz = rz.val
        pose[0, 0] = x
        pose[0, 1] = y
        pose[0, 2] = z
        pose[0, 3] = vrx
        pose[0, 4] = vry
        pose[0, 5] = vrz

        new_img = test_warp(timg, depth, pose, intrinsic)
        l.set_data(new_img)
        fig.canvas.draw_idle()


    sx.on_changed(update)
    sy.on_changed(update)
    sz.on_changed(update)
    rx.on_changed(update)
    ry.on_changed(update)
    rz.on_changed(update)

    plt.show()


if __name__ == "__main__":
    batch_size = 5
    w = 10
    h = 10
    img = torch.rand(batch_size, 3, h, w)
    depth = torch.rand(batch_size, h, w)
    intrinsic = torch.rand(batch_size, 3, 3)
    pose = torch.rand(batch_size, 6)


    # ==============================================================================================

    # ==============================================================================================