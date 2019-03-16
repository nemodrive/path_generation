import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import cv2


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform=None, fps_sample=3):
        """
        Args:
            time_diff : (seconds) Dime distance between frames
            csv_path (string): path to csv file
            fps_sample (fps): Max Sample from video
            transform: pytorch transforms for transforms and tensor conversion
        """

        self._time_diff = 1  # Dime distance between frames

        # Transforms
        self.transforms = transform

        # Read the csv file
        self.data_info = data_info = pd.read_csv(csv_path)

        diff_pts = data_info.pts.values[1:] - data_info.pts.values[:-1]
        diff_pts = diff_pts[diff_pts > 0]

        median_diff, mean_diff, std_diff = np.median(diff_pts), diff_pts.mean(), diff_pts.std()

        print(f"Video FPS -  median: {1./median_diff}, mean: {1./mean_diff}, std: {1./std_diff}")
        print(f"____WILL CONSIDER FIX FPS for frame offset calc_____ {1./median_diff}")

        self.dist_frames = median_diff

        # Third column is for pts viedo -> get groups based on fps sample
        bins = np.arange(0, data_info.pts.max(), 1. / fps_sample)
        data_info["interval"] = data_info.pts.apply(pd.cut, bins=bins, include_lowest=True)\
            .values.codes

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

        sample_start_idx = df.groupby(["video", "interval"]).apply(lambda x: x.sample())

        first_idx = sample_start_idx.index.levels[2]
        idx_dist = max(1, int(time_diff / dist_frames))
        second_idx = first_idx + idx_dist

        # Filter idx that does not exist
        has_df = second_idx < len(df)
        first_idx = first_idx[has_df]
        second_idx= second_idx[has_df]

        # filter same video
        same_video = df.loc[first_idx, "video"].values == df.loc[second_idx, "video"].values
        first_idx = first_idx[same_video]
        second_idx = second_idx[same_video]

        self.first_idx = np.asarray(df.loc[first_idx, "img"])
        self.second_idx = np.asarray(df.loc[second_idx, "img"])

        self.data_len = len(first_idx)

    def __getitem__(self, index):
        # Open image
        img_first = cv2.imread(self.first_idx[index])
        img_second = cv2.imread(self.second_idx[index])

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


if __name__ == "__main__":

    dataset = CustomDatasetFromImages("results/")
