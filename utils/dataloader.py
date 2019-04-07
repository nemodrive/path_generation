import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import cv2
from typing import Tuple

IMG_H = "img"
VIDEO_H = "video"
PTS = "pts"


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path: str, img_size: Tuple, transform=None, fps_sample=3):
        """
        Args:
            time_diff : (seconds) Dime distance between frames
            csv_path (string): path to csv file
            fps_sample (fps): Max Sample from video
            transform: pytorch transforms for transforms and tensor conversion
        """

        self._time_diff = 1  # Dime distance between frames
        self._img_size = tuple(img_size)

        # Transforms
        self.transforms = transform

        # Read the csv file
        self.data_info = data_info = pd.read_csv(csv_path)

        diff_pts = data_info[PTS].values[1:] - data_info[PTS].values[:-1]
        diff_pts = diff_pts[diff_pts > 0]

        median_diff, mean_diff, std_diff = np.median(diff_pts), diff_pts.mean(), diff_pts.std()
        print(f"Video FPS -  median: {1./median_diff}, mean: {1./mean_diff}")
        print(f"Video FPS - std: {std_diff} max: {diff_pts.mean()}")
        print(f"____WILL CONSIDER FIX FPS for frame offset calc_____ {1./median_diff}")

        self.dist_frames = median_diff

        # Third column is for pts viedo -> get groups based on fps sample
        bins = np.arange(0, data_info[PTS].max(), 1. / fps_sample)
        interval = data_info.groupby(VIDEO_H)[PTS].apply(pd.cut, bins=bins,
                                                        include_lowest=True).values.codes
        data_info["interval"] = interval

        # Calculate len
        self.data_len = len(self.data_info.index)

        # Img pair file paths
        self.first_idx = None
        self.second_idx = None

        self.set_groups()

    def set_groups(self):
        df = self.data_info
        dist_frames = self.dist_frames
        time_diff = self.time_diff

        print("Get first index ...")
        sample_start_idx = df.groupby([VIDEO_H, "interval"]).apply(lambda x: x.sample())

        print("Select second index ...")
        first_idx = sample_start_idx.index.levels[2]
        idx_dist = max(1, int(time_diff / dist_frames))
        second_idx = first_idx + idx_dist

        print("Filter indexes ...")
        # Filter idx that does not exist
        has_df = second_idx < len(df)
        first_idx = first_idx[has_df]
        second_idx= second_idx[has_df]

        # filter same video
        same_video = df.loc[first_idx, VIDEO_H].values == df.loc[second_idx, VIDEO_H].values
        first_idx = first_idx[same_video]
        second_idx = second_idx[same_video]

        self.first_idx = np.asarray(df.loc[first_idx, IMG_H])
        self.second_idx = np.asarray(df.loc[second_idx, IMG_H])

        self.data_len = len(first_idx)

    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, self._img_size)
        # img = img.transpose((2, 0, 1))
        # img = img.reshape((1,) + img.shape)
        return img

    def __getitem__(self, index):
        # TODO Reset group indexes after each epoch!
        # Open image
        img_first = self.read_img(self.first_idx[index])
        img_second = self.read_img(self.second_idx[index])

        if self.transforms is not None:
            # Transform image to tensor
            img_first = self.transforms(img_first)
            img_second = self.transforms(img_second)

        return img_first, img_second

    def __len__(self):
        return self.data_len

    @property
    def time_diff(self):
        return self._time_diff

    @time_diff.setter
    def time_diff(self, value):
        self._time_diff = value
        self.set_groups()


if __name__ == "__main__":
    # TODO Check config

    dataset = CustomDatasetFromImages("dataset/small_upb.csv")

    for i in range(10):
        img1, img2 = dataset[i]
        cv2.imshow("IMG1", img1)
        cv2.imshow("IMG2", img2)
        cv2.waitKey(0)
