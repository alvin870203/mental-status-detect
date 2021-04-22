import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, f_transform = None, c_transform=None):
        self.dataset = dataset
        self.c_transform = c_transform
        self.f_transform = f_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        context_path, face_path, id = self.dataset[index]
        context = read_image(context_path)
        face    = read_image(face_path)

        if (self.c_transform is not None) and (self.f_transform is not None):
            context = self.c_transform(context)
            face    = self.f_transform(face)

        return context, face, id
