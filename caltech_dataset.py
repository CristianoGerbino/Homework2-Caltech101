from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(root, split, class_to_idx):
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image and lables: (img_path, label)
    """
    dataset = []

    # Our dir names
    split_file = split + '.txt' 

    # Get all the filenames associated with the split, removing the ones associated to background class
    with open(os.path.join(root.split(os.sep)[0], split_file)) as f: 
      split_names = [line.strip() for line in f if not line.strip().lower().startswith('background')]

    for name in sorted(split_names):
      label_name = name.split(os.sep)[0]
      path = os.path.join(root, name)
      item = (path, class_to_idx[label_name])
      dataset.append(item)
    
    return dataset

    


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, self.split, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples


    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir() \
                       and not d.name.lower().startswith('background')]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) \
                       and not d.name.lower().startswith('background')]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
      
        path, label = self.samples[index]
        image = pil_loader(path)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)
