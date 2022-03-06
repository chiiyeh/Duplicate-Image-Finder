from pathlib import Path
from typing import Optional, List
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import collections


def dup_image_search(directory_A: Path, directory_B: Optional[Path] = None, similarity: str = "normal", px_size: int = 50, show_output: bool = False, delete: bool = False, move: bool = False):
    """
    directory_A (Path)......folder path to search for duplicate/similar images
    directory_B (Path)....second folder path to search for duplicate/similar images that exist in directory_A
    similarity (str)....."normal" = searches for duplicates, recommended setting, MSE < 200
                            "high" = serached for exact duplicates, extremly sensitive to details, MSE < 0.1
                            "low" = searches for similar images, MSE < 1000
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

    result, lower_quality = search_dir(directory_A, directory_B, similarity, px_size, show_output, delete)
    if len(lower_quality) != len(set(lower_quality)):
        print("DifPy found that there are duplicates within directory A.")


    time_elapsed = np.round(time.time()-start_time, 4)

    if len(result) == 1:
        images = "image"
    else:
        images = "images"
    print("Found", len(result), images, "with one or more duplicate/similar images in", time_elapsed, "seconds.")

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


def search_dir(directory_A, directory_B, similarity="normal", px_size=50, show_output=False, delete=False) -> tuple[dict[Path, List[Path]], List[Path]]:
    """
    test
    """
    thres = 15
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
    for _, matrix in matrix_A.items():
        dif = np.expand_dims(matrix, axis=1) - matrix_B["normal"]
        dif = abs(dif).sum(2)
        dif = dif / (3*px_size*px_size)
        for i in np.transpose((dif<thres).nonzero()):
            if i[0] != i[1] or directory_A != directory_B:
                dirA_ind_sim.setdefault(i[0], []).append(i[1])
                if show_output:
                    show_img_figs(matrix[i[0]], matrix_B["normal"][i[1]], px_size, dif[(i[0], i[1])])
                    print(f"Duplicate files:\n{filenames_A[i[0]]} and \n{filenames_B[i[1]]}")

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


def create_imgs_matrix(directory: Path, px_size: int, getrotate: bool = False) -> tuple[dict[str, np.ndarray], List[Path]]:
    """
    Read images in directory, downsize to pxsize and store in matrix.
    If getrotate will generate the different rotations 90, 180, 270 degrees.
    """
    img_filenames = []
    # create list of all files in directory
    folder_files = [filename for filename in directory.rglob('*') if filename.is_file()]

    # create images matrix
    img_matrix_rotdict = {
        "normal": [],
        "90": [],
        "180": [],
        "270": []
    }
    for filename in folder_files:
        # check if the file is an image
        try:
            with Image.open(filename) as im:
                im = im.resize((px_size, px_size))
                im = im.convert("RGB")
                img_matrix_rotdict["normal"].append(np.asarray(im).flatten())
                img_filenames.append(filename)
                if getrotate:
                    img_matrix_rotdict["90"].append(np.asarray(im.transpose(Image.ROTATE_90)).flatten())
                    img_matrix_rotdict["180"].append(np.asarray(im.transpose(Image.ROTATE_180)).flatten())
                    img_matrix_rotdict["270"].append(np.asarray(im.transpose(Image.ROTATE_270)).flatten())
        except OSError:
            pass
    matrix = dict()
    matrix["normal"] = np.asarray(img_matrix_rotdict["normal"], dtype=float)
    if getrotate:
        matrix["90"] = np.asarray(img_matrix_rotdict["90"], dtype=float)
        matrix["180"] = np.asarray(img_matrix_rotdict["180"], dtype=float)
        matrix["270"] = np.asarray(img_matrix_rotdict["270"], dtype=float)
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
