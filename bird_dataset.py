import os
import torch
from torchvision import datasets, transforms
from collections import defaultdict
import random
from PIL import Image
import json
from skimage import io, color, metrics, transform


def compute_ssim(img1, img2):
    img1_gray = color.rgb2gray(img1)
    img2_gray = color.rgb2gray(img2)
    return metrics.structural_similarity(img1_gray, img2_gray)


def resize_image(image, target_size=(500, 375)):
    return transform.resize(image, target_size, mode='reflect', anti_aliasing=True)


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
                 should_save_maps=True,
                 ):

        dataset_images_folder = 'processed-images' if use_no_bg_images else 'images'

        img_root = os.path.join(root, dataset_images_folder)
        self.my_img_root = img_root

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

        if should_save_maps and not os.path.exists(os.path.join(root, 'map_idx.json')):
            print(f'Creating Maps in {root}')
            save_map_path = os.path.join(root)
            # Convert dictionaries to JSON-formatted strings
            json_map_idx = json.dumps(map_idx)
            json_map_parent = json.dumps(map_parent)

            # Save JSON strings to files
            with open(f'{save_map_path}/map_idx.json', 'w') as file:
                file.write(json_map_idx)

            with open(f'{save_map_path}/map_parent.json', 'w') as file:
                file.write(json_map_parent)

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
                    id_name = path.split('/')[0]
                    _, name_parent = id_name.split('.')
                    name_split = name_parent.split('_')
                    parent_name = name_split[-1]
                    name = '_'.join(name_split[:-1])
                    # for cases when there is no child, example - bobolink, etc.
                    if name == '':
                        name = parent_name

                    # Update the parent map
                    if name not in map_parent[parent_name]:
                        map_parent[parent_name][name] = []
                    else:
                        map_parent[parent_name][name].append(path)

        # Second pass: process the image paths
        with open(path_to_index, 'r') as in_file:
            for line_fn in in_file:
                idx, path = line_fn.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    # Extract the details from the path
                    id_name = path.split('/')[0]
                    class_id, name_parent = id_name.split('.')
                    name_split = name_parent.split('_')
                    parent_name = name_split[-1]
                    name = '_'.join(name_split[:-1])
                    if name == '':
                        name = parent_name

                    # Update the maps
                    if idx not in map_idx:
                        map_idx[int(idx)] = {
                            'parent_details': {
                                'parent_name': parent_name,
                                'children': list(map_parent[parent_name].keys())
                            },
                            'name': name,
                            'id': class_id,
                            'file_name': path,
                            'others': map_parent[parent_name][name]
                        }
                    # else:
                    #     map_idx[idx]['others'].append(path)
                    #
                    # map_parent[parent_name][name].append(path)

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
        random_value = random.choice(curr_anchor['others'])
        return random_value

    def _get_negative_anchor(self, idx, map_idx, map_parent, positive_anchor_path):
        curr_anchor = map_idx[idx]
        selected_children = curr_anchor['parent_details']['children']
        random_child = random.choice(selected_children)
        new_child = map_parent[curr_anchor['parent_details']['parent_name']][random_child]
        # random_value = random.choice(new_child)

        most_different_image = None
        min_similarity = float('inf')
        positive_anchor = io.imread(f'{self.my_img_root}/{positive_anchor_path}')
        positive_anchor_resized = resize_image(positive_anchor)

        for neg_path in new_child:
            negative_image = io.imread(f'{self.my_img_root}/{neg_path}')
            negative_image_resized = resize_image(negative_image, target_size=positive_anchor_resized.shape[:2])

            similarity = compute_ssim(positive_anchor_resized, negative_image_resized)

            if similarity < min_similarity:
                min_similarity = similarity
                most_different_image = neg_path

        return most_different_image

    def _get_triplet_anchors(self, root, indices_to_use, map_idx, map_parent):
        triplet_anchors = []
        for idx in indices_to_use:
            anchor = map_idx[idx]['file_name']
            positive_anchor = self._get_positive_anchor(idx, map_idx)
            negative_anchor = self._get_negative_anchor(idx, map_idx, map_parent, positive_anchor)
            triplet_anchors.append((anchor, positive_anchor, negative_anchor))
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
            # anchor, positive_anchor, negative_anchor = self.triplet[index]
            _, positive_anchor, negative_anchor = self.triplet[index]
            # anchor_path = f'{self.my_img_root}/{anchor}'
            positive_anchor_path = f'{self.my_img_root}/{positive_anchor}'
            negative_anchor_path = f'{self.my_img_root}/{negative_anchor}'
            # load the image
            # load_anchor = Image.open(anchor_path).convert('RGB')
            load_anchor = sample
            load_positive = Image.open(positive_anchor_path).convert('RGB')
            load_negative = Image.open(negative_anchor_path).convert('RGB')

            if self.transform_ is not None:
                load_anchor = self.transform_(load_anchor)
                load_positive = self.transform_(load_positive)
                load_negative = self.transform_(load_negative)

            # positive_tensor = transforms.ToTensor()(load_positive)
            # negative_tensor = transforms.ToTensor()(load_negative)

            # target = torch.cat((target, positive_tensor.view(-1), negative_tensor.view(-1)))
            triplet.append((load_anchor, load_positive, load_negative))
            # triplet.append((positive_tensor, negative_tensor))

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target, triplet
