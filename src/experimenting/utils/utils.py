import os
import re

import albumentations
import hydra
from omegaconf import DictConfig, ListConfig


def get_files_paths_in_subtree(path, extensions):
    video_files = []
    extension_regex = "|".join(extensions)

    for root, dirs, files in os.walk(path):
        for f in files:
            if re.search(r'({})$'.format(extension_regex), f):
                video_files.append(os.path.join(root, f))

    return sorted(video_files)


def get_file_paths(path, extensions):
    extension_regex = "|".join(extensions)
    print(extension_regex)
    res = [
        os.path.join(path, f) for f in os.listdir(path)
        if re.search(r'({})$'.format(extension_regex), f)
    ]
    return sorted(res)


def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict) or isinstance(val, DictConfig):
            val = [val]
        if isinstance(val, list) or isinstance(val, ListConfig):
            count_element = 0
            for subdict in val:
                if isinstance(subdict, dict) or isinstance(
                        subdict, DictConfig):
                    deeper = flatten(subdict).items()
                    out.update(
                        {key + '__' + key2: val2
                         for key2, val2 in deeper})
                else:
                    out.update(
                        {key + '__list__' + str(count_element): subdict})
                    count_element += 1

        else:
            out[key] = val
    return out


def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split("__")
        d = resultDict
        i = 0
        while (i < len(parts) - 1):
            part = parts[i]
            is_list = False
            if part == "list":
                part = parts[i - 1]
                if not isinstance(previous[part], list):
                    previous[part] = list()
                d = previous[part]
                break

            if part not in d:
                d[part] = dict()

            previous = d
            d = d[part]
            i += 1

        if isinstance(d, list):
            d.append(value)
        else:
            d[parts[-1]] = value
    return DictConfig(resultDict)


def get_augmentation(augmentation_specifics):
    augmentations = []

    output_info = augmentation_specifics.info
    for _, aug_spec in augmentation_specifics.apply.items():
        aug = hydra.utils.instantiate(aug_spec)

        augmentations.append(aug)

    return albumentations.Compose(augmentations), output_info
