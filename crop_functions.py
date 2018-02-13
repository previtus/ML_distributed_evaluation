import os, sys
import numpy as np
import random
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def best_squares_overlap(w, h, horizontal_splits, overlap_px):

    crop = int((h + overlap_px * (horizontal_splits - 1)) / horizontal_splits)

    row_list = []
    for i in range(0, horizontal_splits):
        row_list.append([int(i * (crop - overlap_px)), int(i * (crop - overlap_px) + crop)])

    n_v = math.ceil((w - crop) / (crop - overlap_px) + 1)
    loc = (w - crop) / (n_v - 1)


    column_list = []

    column_list.append([0,crop])
    for i in range(1, n_v - 1):
        column_list.append([int(i*(loc)), int(i*(loc)+crop)])
    column_list.append([w-crop, w])

    #print(len(column_list) * len(row_list))

    return column_list, row_list


def crop_from_one_img(img, horizontal_splits, overlap_px, scale, show=False, save_crops=True, folder_name='', frame_name=''):

    width, height = img.size

    if show:
        fig, ax = plt.subplots()

        plt.imshow(img)
        plt.xlim(-1 * (width / 10.0), width + 1 * (width / 10.0))
        plt.ylim(-1 * (height / 10.0), height + 1 * (height / 10.0))
        plt.gca().invert_yaxis()

    column_list, row_list = best_squares_overlap(width,height,horizontal_splits,overlap_px)
    crop = column_list[0][1] - column_list[0][0]
    w_crops = column_list
    h_crops = row_list
    #print("after w",w_crops)
    #print("after h",h_crops)

    N = len(w_crops) * len(h_crops)

    crops = []
    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            if show:
                jitter = random.uniform(0, 1) * 15

                ax.add_patch(
                    patches.Rectangle(
                        (w_crop[0] + jitter, h_crop[0] + jitter),
                        scale * crop,
                        scale * crop, fill=False, linewidth=2.0, color=np.random.rand(3, 1)  # color=cmap(i)
                    )
                )

            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            if save_crops:
                file_name = frame_name + str(i).zfill(4) + ".jpg"
                if not os.path.exists(folder_name + frame_name + "/"):
                    os.makedirs(folder_name + frame_name + "/")
                cropped_img.save(folder_name + file_name)
                crops.append((file_name, area))
            else:
                crops.append((None,area))
            i += 1

    if show:
        plt.show()

    return crops, crop

#@profile
def mask_from_one_image(frame_image, SETTINGS, mask_folder):
    ow, oh = frame_image.size

    horizontal_splits = SETTINGS["attention_horizontal_splits"]
    overlap_px = 0
    crop = 608 * horizontal_splits

    nh = crop
    scale_full_img = nh / oh
    nw = ow * scale_full_img

    tmp = frame_image.resize((int(nw), int(nh)), Image.ANTIALIAS)

    save_crops = SETTINGS["debug_save_crops"]

    mask_crops, crop = crop_from_one_img(tmp, horizontal_splits, overlap_px, 1.0, folder_name=mask_folder, frame_name="", save_crops=save_crops)

    return mask_crops, scale_full_img, crop

