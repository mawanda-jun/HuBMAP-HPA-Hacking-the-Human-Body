import numpy as np

def windows(img_height, img_width, crop_height, crop_width, stride_height, stride_width):
    """
    Get windows from dataset, given the crop dimensions and strides which initialized the cropper. 
    Windows will be made later with the pertinent representation.
    """
    # Calculate padding for height and width. Cropping is not always exact, so we calculate
    # the exact amount of padding we are taking in consideration.
    height_pad = int(crop_height + np.ceil((img_height - crop_height) /
                                            stride_height) * stride_height) - img_height
    width_pad = int(crop_width + np.ceil((img_width - crop_width) /
                                            stride_width) * stride_width) - img_width

    # Calculate number of crops along height and width
    crops_per_row = 1 + \
        int((img_height + height_pad - crop_height) / stride_height)
    crops_per_col = 1 + \
        int((img_width + width_pad - crop_width) / stride_width)

    windows = []
    for i in range(crops_per_row):
        for j in range(crops_per_col):
            row_off = i*stride_height
            col_off = j*stride_width
            height = min(crop_height,
                            img_height - i*stride_height)
            width = min(crop_width, img_width -
                        j*stride_width)
            


            windows.append(
                {
                    'row_off': row_off,
                    "col_off": col_off,
                    "width": width,
                    "height": height
                }
            )

    return windows
