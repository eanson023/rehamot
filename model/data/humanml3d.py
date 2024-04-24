import logging
from pathlib import Path

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import Dataset

from .utils import get_split_keyids, load_annotation

logger = logging.getLogger(__name__)


class HumanML3D(Dataset):
    dataname = "HumanML3D Motion-Language"

    def __init__(self, datapath: str,
                 splitpath: str,
                 dataname: str,
                 split: str = "train",
                 min_motion_len: int = 24,
                 progress_bar: bool = True,
                 framerate: float = 20,
                 downsample: bool = True,
                 max_duration: int = 400,
                 data_augment: bool = False,
                 tiny: bool = False, **kwargs):
        self.dataname = dataname
        self.split = split
        self.ids = []
        # only for training
        data_augment = data_augment and self.split == "train"

        super().__init__()
        keyids = get_split_keyids(path=splitpath, split=split + '.txt')

        features_data = {}
        texts_data = {}
        durations = {}

        if progress_bar:
            enumerator = enumerate(track(keyids, f"Loading HumanML3D {split}"))
        else:
            enumerator = enumerate(keyids)

        datapath = Path(datapath)
        bad_num = 0
        sub_num = 0
        aug_num = 0
        more_less_keyids = []
        for _, keyid in enumerator:
            # data augmentation
            if keyid[0] == 'M' and not data_augment:
                continue
            line_split = keyid.split(':')
            keyid = line_split[0]
            text_ids = [int(x) for x in line_split[1].split(',')]
            # 加载已经处理好的数据
            motionpath = datapath / 'new_joint_vecs' / (keyid + '.npy')
            try:
                npydata = np.load(motionpath)
            except:
                logger.error(f"{keyid} can not found.")
                bad_num += 1
                continue
            motion, duration = downsample_frames(
                npydata, downsample=downsample, framerate=framerate)
            if torch.isnan(motion).any().item():
                logger.error(f"{keyid} is a dirty data")
                bad_num += 1
                continue
            if duration > max_duration:
                more_less_keyids.append(keyid)
                bad_num += 1
                continue
            anndata, sub_anndata = load_annotation(
                keyid, text_ids, duration, datapath, framerate)
            if len(anndata) == 0 and not sub_anndata:
                logger.error(f"{keyid} has no annotations")
                bad_num += 1
                continue
            if len(anndata) > 0:
                self.ids += [(keyid, x) for x in range(len(anndata))]
                features_data[keyid] = motion
                texts_data[keyid] = anndata
                durations[keyid] = duration
                if keyid[0] == 'M':
                    aug_num += 1
            # subsequence annotation
            for keyid, data in sub_anndata.items():
                self.ids += [(keyid, 0)]
                motion = motion[data['start']:data['end']]
                features_data[keyid] = motion
                texts_data[keyid] = [data['text']]
                durations[keyid] = len(motion)
                sub_num += 1
                if keyid[0] == 'M':
                    aug_num += 1
        logger.info(
            f"{more_less_keyids} are ignored since too much frames")
        percentage = 100 * bad_num / (len(features_data) + bad_num)
        logger.info(
            f"There are {bad_num} sequences ignored ({percentage:.4}%) since no annotation or dirty data or the number of frames is too long")
        percentage = 100 * sub_num / (len(features_data) + sub_num)
        logger.info(
            f"There are {sub_num} ({percentage:.4}%) new sub-motion pairs")
        percentage = 100 * aug_num / (len(features_data) + aug_num)
        logger.info(
            f"There are {aug_num} ({percentage:.4}%) data using data augmentation (include new sub-motion pairs)")

        self.mean = torch.from_numpy(np.load(datapath / 'Mean.npy')).float()
        self.std = torch.from_numpy(np.load(datapath / 'Std.npy')).float()

        self.features_data = features_data
        self.texts_data = texts_data

        self._num_frames_in_sequence = durations
        self.nfeats = len(self[0]["motion"][0])

        self._num_motions = len(features_data)
        self._num_texts = len(self.ids)

        logger.info(
            f"Split: {split} Motions: {self._num_motions} Texts: {self._num_texts}")

    def __getitem__(self, index):
        keyid, cap_idx = self.ids[index]
        motion = self.features_data[keyid]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        element = {"index": index, "motion": self.features_data[keyid], "text": self.texts_data[keyid][cap_idx],
                   "length": self._num_frames_in_sequence[keyid], "keyid": keyid, "text_index": cap_idx}
        return element

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"


def downsample_frames(features, *, downsample, framerate):
    nframes_total = len(features)
    last_framerate = 20

    if downsample:
        step = int(last_framerate / framerate)
        assert step >= 1
        frames = np.arange(0, nframes_total, step)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)
    features = torch.from_numpy(features[frames]).float()
    return features, duration
