# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# %%
from difPy.dif import dup_image_search

# %%
directory = Path("demo_images")

# %%
dup_image_search(directory, move=True)

# %%
folder_files = [filename for filename in directory.rglob('*') if filename.is_file()]

# %%
folder_files

# %%
px_size = 50
getrotate = True
img_filenames = []
img_matrix_rotdict = {
            "normal": [],
            "90": [],
            "180": [],
            "270": []
        }
for filename in folder_files:
    # check if the file is an image
    with Image.open(filename) as im:
        im = im.resize((px_size, px_size))
        im = im.convert("RGB")
        img_matrix_rotdict["normal"].append(np.asarray(im).flatten())
        img_filenames.append(filename)
        if getrotate:
            img_matrix_rotdict["90"].append(np.asarray(im.transpose(Image.ROTATE_90)).flatten())
            img_matrix_rotdict["180"].append(np.asarray(im.transpose(Image.ROTATE_180)).flatten())
            img_matrix_rotdict["270"].append(np.asarray(im.transpose(Image.ROTATE_270)).flatten())


# %%
img_matrix_rotdict["90"][2]

# %%
matrix = dict()
matrix["normal"] = np.asarray(img_matrix_rotdict["normal"], dtype=float)

# %%
matrix["normal"]

# %%
new_matrix = np.asarray(imgs_matrix + imgs_matrix)
new_matrix = new_matrix.astype(float)

# %%
new_matrix.shape

# %%
tmp = np.expand_dims(matrix["normal"], axis=1) - matrix["normal"]

# %%
tmp.shape

# %%
tmp2 = abs(tmp).sum(2)

# %%
np.asarray([[2, 3]])

# %%
tmp2

# %%
tmp2[0,1]

# %%
for i in np.transpose((tmp2 < 100).nonzero()):
    print(i[0])

# %%
np.append(tmp2, np.asarray([[2, 3]]))

# %%
img_filenames

# %%
np.asanyarray(imgs_matrix)

# %%
plt.imshow(im)

# %%
im1 = np.asarray(im,dtype=float)

# %%
tmp = im1.flatten()

# %%
plt.imshow(tmp.reshape(px_size, px_size, 3).astype(np.uint8))

# %%
(im - tmp.reshape(px_size, px_size, 3)).shape

# %%
