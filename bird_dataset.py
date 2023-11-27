import os
import torch
from torchvision import datasets, transforms
from collections import defaultdict
import random
from PIL import Image


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
                 use_triplet=False,
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
        filenames_to_use, map_idx, map_parent = self._get_filenames_to_use(root, indices_to_use)

        print(map_idx, map_parent)

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

        if use_triplet:
            self.triplet = self._get_triplet_anchors(root, indices_to_use, map_idx, map_parent)
        else:
            self.triplet = None

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
        """
           map[idx] = {
               parent_details: {
                   parent_name: 'Albatross',
                   # because Albatross is same
                   children: ['Laysan_Albatross', 'Sooty_Albatross']
               },
               name: 'Black_footed',
               id: '001',
               file_name: '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
               # contains other black_footed images expect the 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
               others: ['001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg', '001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg']
           }
           map['Albatross'] = { map['Black_footed'] = [ <contains_all_image_paths_of_black_footed> ], map['Laysan_Albatross'] = [ <contains_all_image_paths_of_laysan> ], ... }
       """
        map_idx = {}
        map_parent = defaultdict(lambda: defaultdict(list))

        # First pass: collect all children for each parent
        with open(path_to_index, 'r') as in_file:
            for line_fn in in_file:
                idx, path = line_fn.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(path)
                    # Extract the details from the path
                    id_name, file_name = path.split('/')
                    id, name_parent = id_name.split('.')
                    name, parent_name = name_parent.split('_')

                    # Update the parent map
                    if name not in map_parent[parent_name]:
                        map_parent[parent_name][name] = []

        # Second pass: process the image paths
        with open(path_to_index, 'r') as in_file:
            for line_fn in in_file:
                idx, path = line_fn.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    # Extract the details from the path
                    id_name, file_name = path.split('/')
                    id, name_parent = id_name.split('.')
                    name, parent_name = name_parent.split('_')

                    # Update the maps
                    if idx not in map_idx:
                        map_idx[idx] = {
                            'parent_details': {
                                'parent_name': parent_name,
                                'children': list(map_parent[parent_name].keys())
                            },
                            'name': name,
                            'id': id,
                            'file_name': path,
                            'others': []
                        }
                    else:
                        map_idx[idx]['others'].append(path)

                    map_parent[parent_name][name].append(path)

        return filenames_to_use, map_idx, map_parent

    def _get_bounding_boxes(self, root, indices_to_use):
        path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
        bounding_boxes_dl = []

        with open(path_to_bboxes, 'r') as in_file:
            for line_dl in in_file:
                idx, x_dl, y_dl, w_dl, h_dl = map(float, line_dl.strip('\n').split(' '))
                if int(idx) in indices_to_use:
                    bounding_boxes_dl.append((x_dl, y_dl, w_dl, h_dl))

        return bounding_boxes_dl

    def _get_positive_anchor(self, idx, map_idx):
        curr_anchor = map_idx[idx]
        random_value = random.choice(curr_anchor.other)
        return random_value

    def _get_negative_anchor(self, idx, map_idx, map_parent):
        curr_anchor = map_idx[idx]
        random_child = random.choice(curr_anchor.parent_details.children)
        new_child = map_parent[random_child]
        return random.choice(new_child)

    def _get_triplet_anchors(self, root, indices_to_use, map_idx, map_parent):
        triplet_anchors = []
        for idx in indices_to_use:
            positive_anchor = self._get_positive_anchor(idx, map_idx)
            negative_anchor = self._get_negative_anchor(idx, map_idx, map_parent)
            triplet_anchors.append((positive_anchor, negative_anchor))
        return triplet_anchors

    def __getitem__(self, index):
        # Generate one sample
        sample, target = super(BirdsDataset, self).__getitem__(index)
        triplet = []

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

        if self.triplet is not None:
            positive_anchor, negative_anchor = self.triplet[index]
            # load the image
            load_positive = Image.open(positive_anchor).convert('RGB')
            load_negative = Image.open(negative_anchor).convert('RGB')

            if self.transform_ is not None:
                load_positive = self.transform_(load_positive)
                load_negative = self.transform_(load_negative)

            triplet = torch.tensor([load_positive, load_negative])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target, triplet
