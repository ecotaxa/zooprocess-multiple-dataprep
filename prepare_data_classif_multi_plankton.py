import os

import pandas as pd
import numpy as np
from matplotlib import image as mpimg
from PIL import Image
import json



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

def split_files(source_dir, hashtag, ratio=[0.8,0.1,0.1]):
    import shutil
    os.makedirs(os.path.join(source_dir, "train", hashtag), exist_ok=True)
    os.makedirs(os.path.join(source_dir, "test", hashtag), exist_ok=True)
    os.makedirs(os.path.join(source_dir, "valid", hashtag), exist_ok=True)
    files = os.listdir(os.path.join(source_dir,hashtag))
    np.random.shuffle(files)
    split_index_1 = int(len(files) * ratio[1])
    split_index_2 = int(len(files) * ratio[2]) + split_index_1
    test_files = files[:split_index_1]
    valid_files = files[split_index_1:split_index_2]
    train_files = files[split_index_2:]
    for f in train_files:
        shutil.move(os.path.join(source_dir, hashtag, f), os.path.join(source_dir, "train", hashtag, f))

    for f in test_files:
        shutil.move(os.path.join(source_dir, hashtag, f), os.path.join(source_dir, "test", hashtag, f))

    for f in valid_files:
        shutil.move(os.path.join(source_dir, hashtag, f), os.path.join(source_dir, "valid", hashtag, f))

    print(f"photos {hashtag} split into train,test and valid sets whith ratios {ratio}.")

    shutil.rmtree(os.path.join(source_dir,hashtag))

    return


def prepare_data_classif(sep_img_folder_path,save_folder,no_sep_img_folder_path, 
                         limit_per_scan=20, nb_max_multi=0,nb_max_solo=0, ratio=[0.8,0.1,0.1]):
    count = 1
    count_img_total = 0
    list_scans_sep = get_list_scan(sep_img_folder_path)
    save_folder_multi = os.path.join(save_folder, "multi_plankton")
    os.makedirs(save_folder_multi, exist_ok=True)
    save_folder_solo = os.path.join(save_folder, "solo_plankton")

    os.makedirs(save_folder_solo, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "test"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "valid"), exist_ok=True)


    #Get metadat for images with separation
    sep_metadata_df = get_metadata(list_scans_sep, sep_img_folder_path)

    nb_solo = 0
    nb_multi = 0

    # Get metadata for images without separation
    list_scans_no_sep = get_list_scan(no_sep_img_folder_path)

    no_sep_metadata_df = get_metadata(list_scans_no_sep,no_sep_img_folder_path)


    for i, scan_id in enumerate(list_scans_sep):
        count_per_scan = 0

        if nb_max_solo and nb_solo >= nb_max_solo and nb_max_multi and nb_multi >= nb_max_multi:
            print("We reach the number {} of solo and {} of multi plankton, stopping".format(nb_max_solo,nb_max_multi))
            break

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

            if len(sub_elems) > 1:
                if nb_multi >= nb_max_multi and nb_max_multi:
                    print("Skipping {} because we reach the number {} of multi plankton".format(elem_no_sep.object_id,nb_max_multi))
                    count += 1
                    continue
                else:
                    nb_multi += 1
                    img_no_sep_path = "{}/{}/{}.jpg".format(no_sep_img_folder_path, scan_id,
                                                            elem_no_sep.object_id)
                    img_no_sep = mpimg.imread(img_no_sep_path)

                    group_image = Image.fromarray(img_no_sep)
                    group_image_path = "{}/img_{:05}.png".format(save_folder_multi, count)
                    group_image.save(group_image_path)
                    count += 1
                    continue


            count_per_scan += 1
        
            #print("\t", elem_no_sep[["object_id"]][0])

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
                                     ((pad_y_min, pad_y_max), (pad_x_min, pad_x_max))).astype(int)

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
                masks[padded_mask == 1] = 255 - (k + 1)
                n_masks += 1

            if n_masks == 1:
                if nb_solo >= nb_max_solo and nb_max_solo:
                    print("Skipping {} because it only contains one plankton and we reach the \
                          number {} of solo plankton".format(elem_no_sep.object_id,nb_max_solo))
                    count += 1
                    continue
                else:
                    nb_solo += 1
                    count_per_scan += 1
                    count_img_total += 1
                    group_image = Image.fromarray(img_no_sep)
                    group_image_path = "{}/img_solo_{:05}.png".format(save_folder_solo, count)
                    group_image.save(group_image_path)

                    count += 1

            elif n_masks > 1:
                if nb_multi >= nb_max_multi and nb_max_multi:
                    print("Skipping {} because we reach the number {} of multi plankton".format(elem_no_sep.object_id,nb_max_solo))
                    count += 1
                    continue
                else:
                    nb_multi += 1
                    group_image = Image.fromarray(img_no_sep)
                    group_image_path = "{}/img_solo_{:05}.png".format(save_folder_multi, count)
                    group_image.save(group_image_path)
                    count += 1
            # print("\t", elem_no_sep[["object_id"]][0])

            # Process and save image of plankton group
            # img_no_sep_path = "{}/{}/{}.jpg".format(no_sep_img_folder_path, scan_id, elem_no_sep.object_id)
            # img_no_sep = mpimg.imread(img_no_sep_path)

            # count += 1
            # count_img_total += 1

        print(count)

    print("Total number of images saved: {}".format(count_img_total))


    split_files(save_folder, "multi_plankton", ratio)
    split_files(save_folder, "solo_plankton", ratio)

    return

if __name__ == "__main__":
    import argparse
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
    parser.add_argument("--max_nb_solo", action="store", type=int, dest="max_nb_solo", default=0,
                        help="number max of images of solo plankton to keep in the dataset")
    parser.add_argument("--max_nb_multi", action="store", type=int, dest="max_nb_multi", default=0,
                        help="number max of images of multiple plankton to keep in the dataset")
    parser.add_argument("--ratio", action="store", type=list, dest="ratio", default=[0.1,0.1,0.9], help="ratio (for train) to split train/test")

    args = parser.parse_args()
    save_folder = args.save_folder
    if not save_folder:
        save_folder = "/home/lovnower/learn_data/classif_multi_plankton"

    # Folder path to images with separation, containing our masks
    if args.sep_path:
        sep_img_folder_path = args.sep_path
    else:
        sep_img_folder_path = "/remote/complex/piqv/plankton_ro/zooscan_archives_tara/Zooscan_tara_oceans_2009_2013_wp2_200_with_more_sepMSK/Zooscan_scan/_work"
        print(f"on utilise le dossier par defaut pour les images avec separation : {sep_img_folder_path}")

    # Folder path to images without separation, containing our pictures with grouped plankton
    if args.no_sep_path:
        no_sep_img_folder_path = args.no_sep_path
    else:
        no_sep_img_folder_path = "/remote/complex/piqv/plankton_ro/zooscan_archives_tara/Zooscan_tara_oceans_2009_2013_wp2_200_without_sepMSK/Zooscan_scan/_work"
        print(f"on utilise le dossier par defaut pour les images sans separation : {no_sep_img_folder_path}")


    if not os.path.exists(sep_img_folder_path):
        print(f"Error: The folder {sep_img_folder_path} does not exist.")
        exit(1)
    if not os.path.exists(no_sep_img_folder_path):
        print(f"Error: The folder {no_sep_img_folder_path} does not exist.")
        exit(1)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    

    prepare_data_classif(sep_img_folder_path, save_folder, no_sep_img_folder_path, 
                         nb_max_multi=args.max_nb_multi, nb_max_solo=args.max_nb_solo, 
                         ratio=args.ratio, limit_per_scan=args.limit_per_scan)


    print(f"Data preparation completed. Check the save folder: {save_folder}")


