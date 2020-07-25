import numpy as np

from ..utils import get_file_paths, get_frame_info


class BaseDatasetParams:
    def __init__(self, hparams_dataset):
        self.hparams_dataset = hparams_dataset
        self._set()

        if hparams_dataset.save_split:
            self._save_params(hparams_dataset.preload_dir)

    def _set(self):
        pass

    def _set_train_test_split(self, split_at):

        data_indexes = np.arange(len(self.file_paths))
        cond = self.get_partition_function()
        test_subject_indexes_mask = [cond(x) for x in self.file_paths]

        self.test_indexes = data_indexes[test_subject_indexes_mask]
        data_index = data_indexes[~np.in1d(data_indexes, self.test_indexes)]
        self.train_indexes, self.val_indexes = _split_set(data_index,
                                                          split_at=split_at)

    def get_partition_function(self):
        pass


class DHP19Params(BaseDatasetParams):
    def __init__(self, hparams_dataset):
        super(DHP19Params, self).__init__(hparams_dataset)

    def _set(self):
        self.file_paths = DHP19Params._get_file_paths_with_cam(
            self.hparams_dataset.data_dir, self.hparams_dataset.cams)

        if self.hparams_dataset.test_subjects is None:
            self.subjects = [1, 2, 3, 4, 5]
        else:
            self.subjects = self.hparams_dataset.test_subjects

        if self.hparams_dataset.movements is None or self.hparams_dataset.movements == 'all':
            self.movements = range(1, 34)
        else:
            self.movements = self.hparams_dataset.movements

        self._set_train_test_split(self.hparams_dataset.split_at)

    def get_partition_function(self):
        return lambda x: get_frame_info(x)[
            'subject'] in self.subjects and get_frame_info(x)[
                'mov'] in self.movements

    def _get_file_paths_with_cam(data_dir, cams=None):

        if cams is None:
            cams = [3]

        file_paths = np.array(
            get_file_paths(data_dir, extensions=['.npy', '.mat']))
        cam_mask = [get_frame_info(x)['cam'] in cams for x in file_paths]

        file_paths = file_paths[cam_mask]

        return file_paths


def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes
