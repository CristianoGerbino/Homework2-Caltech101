{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "caltech_dataset.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMRYQB8oZlRHp2xIntALUnc",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/CristianoGerbino/Homework2-Caltech101/blob/master/caltech_dataset.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uZCuonri5epf",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from torchvision.datasets import VisionDataset\n",
        "\n",
        "from PIL import Image\n",
        "\n",
        "import os\n",
        "import os.path\n",
        "import sys\n",
        "\n",
        "def pil_loader(path):\n",
        "    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)\n",
        "    with open(path, 'rb') as f:\n",
        "        img = Image.open(f)\n",
        "        return img.convert('RGB')\n",
        "\n",
        "\n",
        "def make_dataset(root, split, class_to_idx):\n",
        "    \"\"\"Reads a directory with data.\n",
        "    Returns a dataset as a list of tuples of paired image and lables: (img_path, label)\n",
        "    \"\"\"\n",
        "    dataset = []\n",
        "\n",
        "    # Our dir names\n",
        "    split_file = split + '.txt' \n",
        "\n",
        "    # Get all the filenames associated with the split, removing the ones associated to background class\n",
        "    with open(os.path.join(root.split(os.sep)[0], split_file)) as f: \n",
        "      split_names = [line.strip() for line in f if not line.strip().lower().startswith('background')]\n",
        "\n",
        "    for name in sorted(split_names):\n",
        "      label_name = name.split(os.sep)[0]\n",
        "      path = os.path.join(root, name)\n",
        "      item = (path, class_to_idx[label_name])\n",
        "      dataset.append(item)\n",
        "    \n",
        "    return dataset\n",
        "\n",
        "    \n",
        "\n",
        "\n",
        "class Caltech(VisionDataset):\n",
        "    def __init__(self, root, split='train', transform=None, target_transform=None):\n",
        "        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)\n",
        "\n",
        "        self.split = split # This defines the split you are going to use\n",
        "                           # (split files are called 'train.txt' and 'test.txt')\n",
        "        \n",
        "        classes, class_to_idx = self._find_classes(self.root)\n",
        "        samples = make_dataset(self.root, self.split, class_to_idx)\n",
        "        self.classes = classes\n",
        "        self.class_to_idx = class_to_idx\n",
        "        self.samples = samples\n",
        "\n",
        "\n",
        "    def _find_classes(self, dir):\n",
        "        if sys.version_info >= (3, 5):\n",
        "            # Faster and available in Python 3.5 and above\n",
        "            classes = [d.name for d in os.scandir(dir) if d.is_dir() \\\n",
        "                       and not d.name.lower().startswith('background')]\n",
        "        else:\n",
        "            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) \\\n",
        "                       and not d.name.lower().startswith('background')]\n",
        "        classes.sort()\n",
        "        class_to_idx = {classes[i]: i for i in range(len(classes))}\n",
        "        return classes, class_to_idx\n",
        "\n",
        "    def __getitem__(self, index):\n",
        "      \n",
        "        path, label = self.samples[index]\n",
        "        image = pil_loader(path)\n",
        "\n",
        "        # Applies preprocessing when accessing the image\n",
        "        if self.transform is not None:\n",
        "            image = self.transform(image)\n",
        "\n",
        "        return image, label\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.samples)"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}