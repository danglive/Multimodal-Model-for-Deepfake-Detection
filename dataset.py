from __future__ import annotations
import traceback
import numpy as np
import torch
import torch.utils.data
from utils import log

# Mean and std of latency features (pre-computed)
latency_mean = torch.tensor([
    13.19363815, 13.19501656, 13.18659362, 13.17640294, 13.17066425, 13.17019671, 13.17482536,
    13.17799065, 13.18566418, 13.20933517, 13.25752388, 13.28976352, 13.20823131, 12.90568457, 
    12.38983396, 11.89045171, 11.7443412 , 12.03197746, 12.4989344 , 12.87843759, 13.07726926, 
    13.13755761, 13.14001962, 13.13205015, 13.12779171, 13.12155903, 13.11000457, 13.1013639 , 
    13.10253977, 13.10916571, 13.10936431, -0.66052139, 2.08751058
])
latency_std  = torch.tensor([
    1.22018219, 1.21550165, 1.21229337, 1.20948145, 1.21118108,
    1.22301931, 1.23717952, 1.24618324, 1.24850914, 1.25898727,
    1.29470179, 1.34988986, 1.4100049 , 1.47685704, 1.56634314,
    1.62855303, 1.59250148, 1.54371076, 1.53380911, 1.47927914,
    1.39061644, 1.31827097, 1.26728097, 1.2379907 , 1.22404027,
    1.21914667, 1.21814202, 1.21650715, 1.21238232, 1.20961766,
    1.21228655, 4.35032996, 1.24566509
])

class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading preprocessed data (faces + audio features).
    Each sample includes:
        - video: 3D tensor of shape (C,T,H,W)
        - label: binary label (0 or 1)
        - weight: sample weight
        - latency: precomputed latency features for SyncNet alignment
    """
    def __init__(self, data_frame, transform, augmentation=None):
        self._data_frame = data_frame
        self._transform = transform
        self._augmentation = augmentation

    def __len__(self):
        return len(self._data_frame)

    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        filepath = data_row["filepath"]
        weight = data_row["weight"]
        filename = data_row["filename"].split(".")[0]
        label = int(data_row["label"])

        try:
            # Load .npz file which contains faces and mel spectrogram frames
            metadata = np.load(f"{filepath}/{filename}.npz")
            faces = metadata["faces"]

            # Apply augmentation on face frames if available
            if self._augmentation is not None:
                faces = np.array([self._augmentation(image=face)["image"] for face in faces])

            # Concatenate face frames and mel spectrogram frames along temporal dimension
            frames = np.concatenate([faces, metadata["mel_3cs"]], axis=0)

            # Latency features (mdist, offset, conf)
            mdist, offset, conf = metadata["mdist"], metadata["latency"], metadata["conf"]
            if len(mdist) > 0:
                latency = torch.tensor(np.array(list(mdist) + [offset] + [conf]))
                latency = (latency - latency_mean)/latency_std
            else:
                # If no latency data, use zeros
                latency = torch.zeros(33)

        except Exception as e:
            log(e)
            traceback.print_exc()
            # In case of error, return a zero sample (you could handle differently)
            frames = np.zeros((1, image_size, image_size, 3))
            latency = torch.zeros(33)

        sample_dict = {
            "video": torch.tensor(frames).permute(3, 0, 1, 2).half(), # (C,T,H,W)
            "label": label,
            "weight": weight,
            "video_index": video_index,
            "file_path": filename,
            "latency": latency.half(),
        }

        # Apply normalization transform
        sample_dict = self._transform(sample_dict)
        return sample_dict