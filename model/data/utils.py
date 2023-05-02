import codecs as cs
from pathlib import Path

import numpy as np
import torch.utils.data as data
from hydra.utils import instantiate
from omegaconf import DictConfig

from model.data.tools import collate_datastruct_and_text_v2


def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")


def load_annotation(keyid, text_ids, original_duration, datapath, framerate):
    metapath = datapath / 'texts' / (keyid + ".txt")
    anndata = []
    sub_anndata = {}
    iter_ = -1
    with cs.open(metapath, encoding="utf-8") as f:
        for line in f.readlines():
            iter_ += 1
            # 部分数据增强
            if iter_ not in text_ids:
                continue
            try:
                line_split = line.strip().split('#')
                caption = line_split[0]
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                if f_tag == 0.0 and to_tag == 0.0:
                    anndata.append(caption)
                else:
                    start = int(f_tag * framerate)
                    end = int(to_tag * framerate)
                    # 拆除错误时间帧标注
                    if (end - start) == original_duration:
                        anndata.append(caption)
                        continue
                    new_name = keyid + '_' + str(iter_)
                    sub_anndata[new_name] = {'start': int(f_tag * framerate),
                                             'end': int(to_tag * framerate),
                                             'text': caption}
            except:
                print(f"The data is wrong, it may be caused by the original data wrong line break, please check keyid:{keyid} line_split:{line_split}")

    return anndata, sub_anndata


def get_loader_single(dataname, split, batch_size, workers, shuffle, drop_last, data_cfg: DictConfig):
    """Returns torch.utils.data.DataLoader for dataset."""

    dataset = instantiate(data_cfg, split=split)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=workers,
                                  drop_last=drop_last,
                                  collate_fn=collate_datastruct_and_text_v2)
    return data_loader


def get_loaders(dataname, batch_size, workers, cfg: DictConfig):
    train_loader = get_loader_single(
        dataname, 'train', batch_size, workers, shuffle=True, drop_last=False, data_cfg=cfg)
    val_loader = get_loader_single(
        dataname, 'val', batch_size, workers, shuffle=False, drop_last=False, data_cfg=cfg)
    return train_loader, val_loader
