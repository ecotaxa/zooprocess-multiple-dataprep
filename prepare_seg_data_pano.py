import os
import argparse

import pandas as pd
import numpy as np
from matplotlib import image as mpimg
from PIL import Image
import json
from transformers.image_transforms import id_to_rgb
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
Image.MAX_IMAGE_PIXELS = 2000000000




# Utils functions

def contains(obj, list_obj):
    """
    Check for each object in list_obj if it is contained in obj,
    i.e. if the bounding box of obj entirely covers the bounding box of each object in list_obj

    Parameters:
        obj: A pandas Series
        list_obj: A pandas DataFrame with n rows

    Returns:
        A boolean pandas Series of length n
    """
    # NB: the attributes of obj are str
    return (
            (float(obj["object_bx"]) <= list_obj["object_bx"].astype(float))
            & (float(obj["object_by"]) <= list_obj["object_by"].astype(float))
            & (float(obj["object_bx"]) + float(obj["object_width"]) >= list_obj["object_bx"].astype(float) + list_obj[
        "object_width"].astype(float))
            & (float(obj["object_by"]) + float(obj["object_height"]) >= list_obj["object_by"].astype(float) + list_obj[
        "object_height"].astype(float))
    )


def rgb2gray(rgb):
    """
    Convert an RGB image of shape (h, w, c) image to a grayscale image of shape (h, w)
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def bounding_box(img):
    """
    Get the corners of the bounding box of an object on a white background
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def get_list_scan(img_folder_path):
    # Get metadata for images with separation
    return os.listdir(img_folder_path)


def get_metadata(list_scans, img_folder_path):
    #Get dataframe with metadata for images with & without separation
    metadata_df = pd.DataFrame()
    for i, scan_id in enumerate(list_scans):
        # print(scan_id)
        metadata_path = "{}/{}/ecotaxa_{}.tsv".format(img_folder_path, scan_id, scan_id)
        try:
            df = pd.read_csv(metadata_path, sep="\t")
        except UnicodeDecodeError:
            print("UnicodeDecodeError for:\n", metadata_path)
            continue

        if i == 0:
            metadata_df = df
        else:
            metadata_df = pd.concat([metadata_df, df], axis=0)
    return metadata_df


def prepare_data_pano(sep_img_folder_path,save_folder,no_sep_img_folder_path, 
                      limit_per_scan=20, only_multiple=1,nb_solo_max=0, verbose=False):
    count = 1
    count_img_total = 0
    if only_multiple:
        print("we only keep images with multiple plankton")
    else:
        print("we keep all images, including images with a single plankton")
        if nb_solo_max:
            print(f"we keep at most {nb_solo_max} images with a single plankton")
    list_scans_sep = get_list_scan(sep_img_folder_path)
    images_folder = os.path.join(save_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    masks_folder = os.path.join(save_folder, "labels")
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
    json_folder = os.path.join(save_folder, "seg_info")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    print("folders images, labels and seg_info created")

    #Get metadat for images with separation
    sep_metadata_df = get_metadata(list_scans_sep, sep_img_folder_path)

    nb_solo = 0

    # Get metadata for images without separation
    list_scans_no_sep = get_list_scan(no_sep_img_folder_path)

    no_sep_metadata_df = get_metadata(list_scans_no_sep,no_sep_img_folder_path)


    mask_count_total = 0
    id_depart = 157268

    for i, scan_id in enumerate(list_scans_sep):
        count_per_scan = 0

        # if i==11: break
        print('{}/{}: {}'.format(i + 1, len(list_scans_sep), scan_id))

        # Get metadata of both sep and no_sep images for a single scan
        sub_df_sep = sep_metadata_df[sep_metadata_df.object_id.str.startswith(scan_id)]
        sub_df_no_sep = no_sep_metadata_df[no_sep_metadata_df.object_id.str.startswith(scan_id)]

        # Loop through "no_sep" objects, i.e. images containing groups of plankton
        for j in range(len(sub_df_no_sep)):
            if count_per_scan >= limit_per_scan:
                continue
            elem_no_sep = sub_df_no_sep.iloc[j]

            # Get "sep" objects contained in elem_no_sep
            sub_elems = sub_df_sep[contains(elem_no_sep, sub_df_sep)]

            # If elem_no_sep only contains one object, then it's already a single plankton
            if len(sub_elems) == 1 and only_multiple:  # > 1 means we ONLY KEEP UNIQUES
                if verbose:
                    print("Skipping {} because it only contains one plankton".format(elem_no_sep.object_id))
                count += 1
                continue
            elif not only_multiple and len(sub_elems) == 1 and nb_solo_max < nb_solo and nb_solo_max:
                if verbose:
                    print("Skipping {} because it only contains one plankton and we reach the \
                      number {} of solo plankton".format(elem_no_sep.object_id,nb_solo_max))
                count += 1
                continue

            # print("\t", elem_no_sep[["object_id"]][0])

            # Process and save image of plankton group
            img_no_sep_path = "{}/{}/{}.jpg".format(no_sep_img_folder_path, scan_id, elem_no_sep.object_id)
            img_no_sep = mpimg.imread(img_no_sep_path)

            threshold = elem_no_sep.process_particle_threshold
            mask = (rgb2gray(img_no_sep) <= float(threshold)).astype(int)
            rmin, rmax, cmin, cmax = bounding_box(mask[:-31])  # Remove the bottom 31 pixels containing the scale bar
            img_no_sep = img_no_sep[rmin:rmax, cmin:cmax]

            # Get each mask with the correct position
            masks = np.zeros(img_no_sep.shape[:2])

            # Keep track of real number of masks
            n_masks = 0
            list_seg_info = list()

            for k in range(len(sub_elems)):
                elem = sub_elems.iloc[k]
                img_path = "{}/{}/{}.jpg".format(sep_img_folder_path, scan_id, elem.object_id)
                img = mpimg.imread(img_path)

                # Compute bounding box
                threshold = elem.process_particle_threshold
                mask = np.array(rgb2gray(img) <= float(threshold)).astype(float)
                rmin, rmax, cmin, cmax = bounding_box(mask[:-31])

                # Compute padding to add to get the correct position on the "no_sep" image
                # NB: The attributes of elem_no_sep are strings, so we need to do multiple float casts
                pad_x_min = int(float(elem.object_bx) - float(elem_no_sep.object_bx))
                pad_x_max = int(float(elem_no_sep.object_bx) + float(elem_no_sep.object_width) - (
                        float(elem.object_bx) + float(elem.object_width)))
                pad_y_min = int(float(elem.object_by) - float(elem_no_sep.object_by))
                pad_y_max = int(float(elem_no_sep.object_by) + float(elem_no_sep.object_height) - (
                        float(elem.object_by) + float(elem.object_height)))
                padded_mask = np.pad(mask[rmin:rmax, cmin:cmax],
                                     ((pad_y_min, pad_y_max), (pad_x_min, pad_x_max))).astype(
                    int)

                # print(masks.shape, padded_mask.shape)

                # Fix rounding errors, not the best approach but will do for now
                if padded_mask.shape[0] == masks.shape[0] - 1:
                    padded_mask = np.concatenate([padded_mask, np.zeros((1, padded_mask.shape[1]))], axis=0)
                    print("added 1 row to padded mask at count {}".format(count))
                if padded_mask.shape[1] == masks.shape[1] - 1:
                    padded_mask = np.concatenate([padded_mask, np.zeros((padded_mask.shape[0], 1))], axis=1)
                    print("added 1 column to padded mask at count {}".format(count))
                if padded_mask.shape[0] == masks.shape[0] + 1:
                    padded_mask = padded_mask[:-1, :]
                    print("removed 1 row to padded mask at count {}".format(count))
                if padded_mask.shape[1] == masks.shape[1] + 1:
                    padded_mask = padded_mask[:, :-1]
                    print("removed 1 column to padded mask at count {}".format(count))

                # Check if the sub_elem really belongs to the plankton group
                if (img_no_sep[padded_mask == 1] != 255).astype(int).sum() == 0:
                    print("sub_elem not really part of plankton group removed")
                    continue

                # Add mask to image of all masks
                # print(masks.shape, padded_mask.shape)
                #masks[padded_mask == 1] = 255 - (k + 1)
                masks[padded_mask == 1] = mask_count_total + id_depart
                list_seg_info.append({"id": mask_count_total + id_depart, "category_id": 1})
                n_masks += 1
                mask_count_total += 1

            if n_masks == 1 :
                nb_solo += 1
            group_image = Image.fromarray(img_no_sep)
            group_image_path = "{}/img_{:05}.jpg".format(images_folder, count)
            group_image.save(group_image_path)

            #mask_image = Image.fromarray(masks).convert('L')
            mask_image = id_to_rgb(masks)
            mask_image_path = "{}/img_{:05}.png".format(masks_folder, count)
            #mask_image.save(mask_image_path)
            plt.imsave(mask_image_path, mask_image)

            #voir avec quel id modifier

            list_seg_info.append({"id": 0, "category_id": 0})
            json_image_path = "{}/img_{:05}.json".format(json_folder, count)
            with open(json_image_path, 'w') as fout:
                json.dump(list_seg_info, fout)

            count += 1
            count_img_total += 1

        print(count)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #ajouter le choix des dossiers de lecture et de sauvegarde
    parser.add_argument("--save_folder", action="store", type=str, dest="save_folder", default="",
                        help="path to save the prepared data")
    parser.add_argument("--sep_path", action="store", type=str, dest="sep_path", default="",help="path to the folder containing images with separation")
    parser.add_argument("--no_sep_path", action="store", type=str, dest="no_sep_path", default="",help="path to the folder containing images without separation")
    parser.add_argument("--limit_per_scan", action="store", type=int, dest="limit_per_scan", default=20,
                        help="limit of images per scan")
    parser.add_argument("--only_multiple", action="store", type=int, dest="only_multiple", default=1,
                        help="if 1, only keep images with multiple plankton, if 0, keep all images")
    parser.add_argument("--nb_solo_max", action="store", type=int, dest="nb_solo_max", default=0,
                        help="if only_multiple=0, keep at most nb_solo_max images with a single plankton")
    parser.add_argument("--verbose", action="store_true", dest="verbose", default=False, help="verbose")

    args = parser.parse_args()
    save_folder = args.save_folder
    if not save_folder:
        print("you must provide a folder to save the prepared data, use --save_folder <path_to_folder>")
        exit()
    else:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"created the folder {save_folder}")
        else:
            if os.listdir(save_folder):
                print(f"the folder {save_folder} is not empty, please provide an empty folder to save the prepared data")
                exit()
    print(f"we save the prepared data to {save_folder}")

    # Default folder path to images with separation, containing our masks
    sep_img_folder_path = "/remote/complex/piqv/plankton_ro/zooscan_archives_tara/Zooscan_tara_oceans_2009_2013_wp2_200_with_more_sepMSK/Zooscan_scan/_work"

    # Default folder path to images without separation, containing our pictures with grouped plankton
    no_sep_img_folder_path = "/remote/complex/piqv/plankton_ro/zooscan_archives_tara/Zooscan_tara_oceans_2009_2013_wp2_200_without_sepMSK/Zooscan_scan/_work"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not args.sep_path:
        sep_img_folder_path = "/remote/complex/piqv/plankton_ro/zooscan_archives_tara/Zooscan_tara_oceans_2009_2013_wp2_200_with_more_sepMSK/Zooscan_scan/_work"
        print(f"no sep_path provided, we use the default one: {sep_img_folder_path}")
    elif not os.path.exists(args.sep_path):
        print(f"Error: The folder {args.sep_path} does not exist.")
        exit(1)
    else:
        sep_img_folder_path = args.sep_path
    if not args.no_sep_path:
        no_sep_img_folder_path = "/remote/complex/piqv/plankton_ro/zooscan_archives_tara/Zooscan_tara_oceans_2009_2013_wp2_200_without_sepMSK/Zooscan_scan/_work"
        print(f"no no_sep_path provided, we use the default one: {no_sep_img_folder_path}")
    elif not os.path.exists(args.no_sep_path):
        print(f"Error: The folder {args.no_sep_path} does not exist.")
        exit(1)
    else:
        no_sep_img_folder_path = args.no_sep_path

    prepare_data_pano(sep_img_folder_path,save_folder,no_sep_img_folder_path, args.limit_per_scan, 
                      args.only_multiple,args.nb_solo_max, args.verbose)


    print(f"Data preparation completed. Check the save folder: {save_folder}")

