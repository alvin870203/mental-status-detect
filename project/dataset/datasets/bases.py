import numpy as np


class BaseDataset(object):
    """
    Base class of CAER dataset
    """

    def get_imagedata_info(self, data):
        eids = []
        for _, _, eid in data:
            eids += [eid]
        eids = set(eids)
        num_eids = len(eids)
        num_imgs = len(data)
        return num_eids, num_imgs

    def get_videodata_info(self, data):
        eids = []
        for _, eid in data:
            eids += [eid]
        eids = set(eids)
        num_eids   = len(eids)
        num_videos = len(data)
        return num_eids, num_videos

    def print_dataset_statistics(self):
        raise NotImplementedError

class BaseImageDataset(BaseDataset):
    """
    Base class of image dataset
    """

    def print_dataset_statistics(self, train, test):
        num_train_eids, num_train_imgs = self.get_imagedata_info(train)
        num_test_eids,  num_test_imgs  = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images |")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} |".format(num_train_eids, num_train_imgs))
        print("  test     | {:5d} | {:8d} |".format(num_test_eids, num_test_imgs))
        print("  ----------------------------------------")

class BaseVideoDataset(BaseDataset):
    """
    Base class of video dataset
    """
    def print_dataset_statistics(self, train, validation, test):
        num_train_eids, num_train           = self.get_videodata_info(train)
        num_validation_eids, num_validation = self.get_videodata_info(validation)
        num_test_eids,  num_test            = self.get_videodata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset     | # ids | # videos |")
        print("  ----------------------------------------")
        print("  train      | {:5d} | {:8d} |".format(num_train_eids, num_train))
        print("  validation | {:5d} | {:8d} |".format(num_validation_eids, num_validation))
        print("  test       | {:5d} | {:8d} |".format(num_test_eids, num_test))
        print("  ----------------------------------------")
