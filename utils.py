import torchvision as tv
import torchvision.transforms.functional as TF


def pad(img_to_pad, fill_with=0, size_max=500):
    """
    Pads images to the specified size (height x width).
    Fills up the padded area with value(s) passed to the `fill` parameter.
    """
    pad_height = max(0, size_max - img_to_pad.height)
    pad_width = max(0, size_max - img_to_pad.width)

    pad_top = int(pad_height // 2)
    pad_bottom = int(pad_height - pad_top)
    pad_left = int(pad_width // 2)
    pad_right = int(pad_width - pad_left)

    padding_config = [pad_left, pad_top, pad_right, pad_bottom]

    return TF.pad(img_to_pad, padding_config, fill=fill_with)


# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x_res: int(round(x_res * 256)), (0.485, 0.456, 0.406)))


# pad images to 500 pixels
# max_padding = tv.transforms.Lambda(lambda x_img: pad(x_img, fill_with=fill))

def apply_padding(x_img):
    return pad(x_img, fill_with=fill)


max_padding = tv.transforms.Lambda(apply_padding)