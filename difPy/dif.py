import argparse
import math
from pathlib import Path
import pathlib
from typing import Optional, List, Dict, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import time


def dup_image_search(directory_A: Path, directory_B: Optional[Path] = None, threshold: float = 5, px_size: int = 50, show_output: bool = False, delete: bool = False, move: bool = False):
    """
    directory_A (Path)......folder path to search for duplicate/similar images
    directory_B (Path)....second folder path to search for duplicate/similar images that exist in directory_A
    threshold (float).....average pixel absolute difference
    px_size (int)........recommended not to change default value
                            resize images to px_size height x width (in pixels) before being compared
                            the higher the pixel size, the more computational ressources and time required
    show_output (bool)...False = omits the output and doesn't show found images
                            True = shows duplicate/similar images found in output
    delete (bool)........! please use with care, as this cannot be undone
                            lower resolution duplicate images that were found are automatically deleted

    OUTPUT (set).........a dictionary with the filename of the duplicate images
                            and a set of lower resultion images of all duplicates
    """
    start_time = time.time()

    if directory_B is not None:
        if not directory_B.is_dir():
            raise FileNotFoundError(f"Directory: {directory_B} does not exist")
    else:
        directory_B = directory_A

    if not directory_A.is_dir():
        raise FileNotFoundError(f"Directory: {directory_A} does not exist")

    result, lower_quality = search_dir(directory_A, directory_B, threshold, px_size, show_output, delete)
    if len(lower_quality) != len(set(lower_quality)):
        print("DifPy found that there are duplicates within directory A.")


    time_elapsed = np.round(time.time()-start_time, 4)

    print(f"Found {len(result)} images with one or more duplicate/similar images in {time_elapsed} seconds.")

    if len(result) != 0:
        if delete:
            usr = input("Are you sure you want to delete all lower resolution duplicate images? \nThis cannot be undone. (y/n)")
            if str(usr) == "y":
                delete_imgs(set(lower_quality))
            else:
                print("Image deletion canceled.")
        if move:
            print(f"Moving duplicates to tmp folder in {directory_A}!")
            tmp_dir = directory_A / 'tmp'
            tmp_dir.mkdir()
            for high_qu, low_qual_images in result.items():
                orig_name = high_qu.stem
                new_name = orig_name + high_qu.suffix
                high_qu.rename(tmp_dir / high_qu.name)
                for idx, low_qu in enumerate(low_qual_images):
                    new_name = orig_name + f"_{idx}" + low_qu.suffix
                    print(new_name)
                    low_qu.rename(tmp_dir / new_name)
    return result


def search_dir(directory_A, directory_B, threshold=5, px_size=50, show_output=False, delete=False) -> Tuple[Dict[Path, List[Path]], List[Path]]:
    """
    test
    """
    memsize = 8000
    matrix_A, filenames_A = create_imgs_matrix(directory_A, px_size, getrotate=True)
    matrix_B = {}
    if directory_B == directory_A:
        filenames_B = filenames_A.copy()
        matrix_B["normal"] = matrix_A["normal"].copy()
    else:
        matrix_B, filenames_B = create_imgs_matrix(directory_B, px_size, getrotate=False)
    result = {}
    lower_quality = []

    # find duplicates/similar images within one folder
    dirA_ind_sim = dict()
    loop_size = math.floor(memsize/(sys.getsizeof(matrix_A["normal"])*50/1000000)) * 50 
    if not loop_size:
        loop_size = 1
    print(f"Maximum loop size: {loop_size}")
    num_images_B = matrix_B["normal"].shape[0]
    num_loop = math.ceil(num_images_B/loop_size) ## CHECK
    print(f"Number of loops: {num_loop}")
    for _, matrix in matrix_A.items():
        for i in range(num_loop):
            if directory_A != directory_B:
                dif = np.expand_dims(matrix, axis=1) - matrix_B["normal"][i*loop_size:min((i+1)*loop_size, num_images_B)]
            else:
                dif = np.expand_dims(matrix[i*loop_size:], axis=1) - matrix_B["normal"][i*loop_size:min((i+1)*loop_size, num_images_B)]
            dif = abs(dif).sum(2)
            dif = dif / (3*px_size*px_size)
            for ind in np.transpose((dif<threshold).nonzero()):
                if directory_A != directory_B:
                    a_ind = ind[0]
                    b_ind = ind[1]+i*loop_size
                elif ind[0] != ind[1]:
                    a_ind = ind[0]+i*loop_size
                    b_ind = ind[1]+i*loop_size
                else:
                    continue
                dirA_ind_sim.setdefault(a_ind, []).append(b_ind)
                if show_output:
                    show_img_figs(matrix[a_ind], matrix_B["normal"][b_ind], px_size, dif[(ind[0], ind[1])])
                    print(f"Duplicate files:\n{filenames_A[a_ind]} and \n{filenames_B[b_ind]}")

    if directory_A == directory_B:
        processed_ls = []
        for ind_A, ls_ind_B in dirA_ind_sim.items():
            if filenames_A[ind_A] not in processed_ls:
                to_add_ls = [filenames_B[ind_B] for ind_B in ls_ind_B] + [filenames_A[ind_A]]
                processed_ls += to_add_ls
                high, low_qual_images = check_img_quality(to_add_ls)
                lower_quality += low_qual_images
                result[high] = low_qual_images
    else:
        for ind_A, ls_ind_B in dirA_ind_sim.items():
            to_add_ls = [filenames_B[ind_B] for ind_B in ls_ind_B] + [filenames_A[ind_A]]
            high, low_qual_images = check_img_quality(to_add_ls)
            lower_quality += low_qual_images
            result[high] = low_qual_images

    return result, lower_quality


def create_imgs_matrix(directory: Path, px_size: int, getrotate: bool = False) -> Tuple[Dict[str, np.ndarray], List[Path]]:
    """
    Read images in directory, downsize to pxsize and store in matrix.
    If getrotate will generate the different rotations 90, 180, 270 degrees.
    """
    img_filenames = []
    # create list of all files in directory
    folder_files = [filename for filename in directory.rglob('*') if filename.is_file()]
    num_files = len(folder_files)
    print(f"Processing {num_files} images!")

    # create images matrix
    img_matrix_rotdict = {
        "normal": [],
        "90": [],
        "180": [],
        "270": []
    }
    count = 0
    for filename in folder_files:
        # check if the file is an image
        try:
            with Image.open(filename) as im:
                im = im.resize((px_size, px_size))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                img_matrix_rotdict["normal"].append(np.asarray(im).flatten())
                img_filenames.append(filename)
                if getrotate:
                    img_matrix_rotdict["90"].append(np.asarray(im.transpose(Image.ROTATE_90)).flatten())
                    img_matrix_rotdict["180"].append(np.asarray(im.transpose(Image.ROTATE_180)).flatten())
                    img_matrix_rotdict["270"].append(np.asarray(im.transpose(Image.ROTATE_270)).flatten())
        except OSError:
            pass
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images out of {num_files}!")
    matrix = dict()
    matrix["normal"] = np.asarray(img_matrix_rotdict["normal"], dtype=np.int32)
    if getrotate:
        matrix["90"] = np.asarray(img_matrix_rotdict["90"], dtype=np.int32)
        matrix["180"] = np.asarray(img_matrix_rotdict["180"], dtype=np.int32)
        matrix["270"] = np.asarray(img_matrix_rotdict["270"], dtype=np.int32)
    return matrix, img_filenames

# Function that plots two compared image files and their mse
def show_img_figs(imageA, imageB, px_size, err):
    """
    Shows the two similar images with the err
    """
    fig = plt.figure()
    plt.suptitle("MSE: %.2f" % (err))
    # plot first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA.reshape(px_size, px_size, 3).astype(np.uint8), cmap=plt.cm.gray)
    plt.axis("off")
    # plot second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB.reshape(px_size, px_size, 3).astype(np.uint8), cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()


def check_img_quality(ls_image_path: List[Path]):
    """
    Function for checking the quality of compared images, appends the lower quality image to the list
    """
    max_ind, _ = max(enumerate(ls_image_path), key=lambda p: p[1].stat().st_size)
    high_qual_file = ls_image_path.pop(max_ind)
    return high_qual_file, ls_image_path

def delete_imgs(lower_quality_set: List[Path]):
    """
    Function for deleting the lower quality images that were found after the search
    """
    for file in lower_quality_set:
        print("\nDeletion in progress...")
        deleted = 0
        try:
            file.unlink()
            print("Deleted file:", file)
            deleted += 1
        except:
            print("Could not delete file:", file)
        print("\n***\nDeleted", deleted, "duplicates/similar images.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_a", action="store", type=pathlib.Path, help="Main directory")
    parser.add_argument("--dir-b", action="store", dest="dir_b", type=pathlib.Path, default=None,help="directory to search for images that exist in dir-a")
    parser.add_argument("--px-size", dest="px_size", action="store", type=int, default=50, help="pixel size to reduce the image before comparison")
    parser.add_argument("--threshold", action="store", type=float, default=5, help="threshold for average absolute pixel difference to consider similar")
    parser.add_argument("--show-output", action="store_true", dest="show_output", default=False, help="Show output")
    parser.add_argument("--delete", action="store_true", default=False, help="Delete the duplicate files of lower memory space")
    parser.add_argument("--move", action="store_true", default=False, help="Move the duplicates as well as the high res image to dir_a/tmp folder")
    args = parser.parse_args()
    dup_image_search(args.dir_a, directory_B=args.dir_b, threshold=args.threshold, px_size=args.px_size, show_output=args.show_output, delete=args.delete, move=args.move)
