import os
import torch
from torchvision import datasets, transforms


class BirdsDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset.
    Method BirdsDataset.__getitem__() returns a tuple of an image and its corresponding label.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 use_bounding_boxes=False,
                 use_no_bg_images=False,
                 ):

        dataset_images_folder = 'processed-images' if use_no_bg_images else 'images'

        img_root = os.path.join(root, dataset_images_folder)

        super(BirdsDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.is_training_set = train

        # Obtain sample ids filtered by split
        indices_to_use = self._get_indices_to_use(root)

        # Obtain filenames of images
        filenames_to_use = self._get_filenames_to_use(root, indices_to_use)

        img_paths_cut = {'/'.join(img_path_x.rsplit('/', 2)[-2:]): idx for idx, (img_path_x, lb) in
                         enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if use_bounding_boxes:
            # Get coordinates of a bounding box
            self.bboxes = self._get_bounding_boxes(root, indices_to_use)
        else:
            self.bboxes = None

    def _get_indices_to_use(self, root):
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = []

        with open(path_to_splits, 'r') as in_file:
            for curr_line in in_file:
                idx, use_train = curr_line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.is_training_set:
                    indices_to_use.append(int(idx))

        return indices_to_use

    def _get_filenames_to_use(self, root, indices_to_use):
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()

        with open(path_to_index, 'r') as in_file:
            for line_fn in in_file:
                idx, fn = line_fn.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        return filenames_to_use

    def _get_bounding_boxes(self, root, indices_to_use):
        path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
        bounding_boxes_dl = []

        with open(path_to_bboxes, 'r') as in_file:
            for line_dl in in_file:
                idx, x_dl, y_dl, w_dl, h_dl = map(float, line_dl.strip('\n').split(' '))
                if int(idx) in indices_to_use:
                    bounding_boxes_dl.append((x_dl, y_dl, w_dl, h_dl))

        return bounding_boxes_dl

    def __getitem__(self, index):
        # Generate one sample
        sample, target = super(BirdsDataset, self).__getitem__(index)

        if self.bboxes is not None:
            # Squeeze coordinates of the bounding box to the range [0, 1]
            x_bbx, y_bbx, w_bbx, h_bbx = self.bboxes[index]

            scale_resize = 500 / sample.width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x_bbx / 375
            y_rel = scale_resize_crop * y_bbx / 375
            w_rel = scale_resize_crop * w_bbx / 375
            h_rel = scale_resize_crop * h_bbx / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])
            # target = torch.tensor([target, x_bbx, y_bbx, w_bbx, h_bbx])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target
