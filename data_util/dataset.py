import os
import glob
import pandas as pd
import numpy as np
import pydicom
from scipy.ndimage import zoom


class CTDataset:
    def __init__(self, path, config, in_memory=True):
        self.FIXED_SIZE = 128

        self.config = config

        self.scan_names = pd.read_csv(path)
        self.scan_names = self.scan_names[self.scan_names.columns[1]]

        self.Fixed = config['Fixed']
        self.Moving = config['Moving']
        self._make_dataset()
        if in_memory is True:
            self.load_in_memory()

    def _make_dataset(self):
        fixed_data = []
        moving_data = []

        for entity in self.scan_names:
            # Adding fixed data
            scan_path = os.path.join(self.config["root_data_path"], entity, self.Fixed, "DICOM")
            slices = list(glob.glob(os.path.join(scan_path, "*")))
            if len(slices) == 0:
                continue
            slices.sort(key=lambda x: int(x.split("/")[-1][1:]))
            fixed_data.append(slices)

            # Adding moving data
            scan_path = os.path.join(self.config["root_data_path"], entity, self.Moving, "DICOM")
            slices = list(glob.glob(os.path.join(scan_path, "*")))
            slices.sort(key=lambda x: int(x.split("/")[-1][1:]))
            moving_data.append(slices)

        self.fixed_data = fixed_data
        self.moving_data = moving_data

    def __getitem__(self, index):
        # Image from Fixed
        fixed = []
        for i in range(0, len(self.fixed_data[index])):
            fixed_image = pydicom.dcmread(self.fixed_data[index][i]).pixel_array
            fixed_image[fixed_image < 0] = 0
            fixed_image = fixed_image / 1e3
            fixed_image = fixed_image - 1
            fixed_image = np.expand_dims(fixed_image, 0).astype(np.float)
            fixed.append(fixed_image)
        fixed = np.vstack(fixed)
        fixed = zoom(fixed, 0.5)
        fixed = np.pad(fixed, ((self.FIXED_SIZE - fixed.shape[0], 0), (0, 0), (0, 0)), "constant")

        # Paired image from Moving
        moving = []
        for i in range(0, len(self.moving_data[index])):
            moving_image = pydicom.dcmread(self.moving_data[index][i]).pixel_array
            moving_image[moving_image < 0] = 0
            moving_image = moving_image / 1e3
            moving_image = moving_image - 1
            moving_image = np.expand_dims(moving_image, 0).astype(np.float)
            moving.append(moving_image)
        moving = np.vstack(moving)
        moving = zoom(moving, 0.5)
        moving = np.pad(moving, ((self.FIXED_SIZE - moving.shape[0], 0), (0, 0), (0, 0)), "constant")

        moving = np.expand_dims(moving, 0)
        fixed = np.expand_dims(fixed, 0)
        return fixed, moving

    def __len__(self):
        return len(self.fixed_data)

    def load_in_memory(self):
        pass
