import torch
from torch import nn

devie = 'cuda' if torch.cuda.is_available() else 'cpu'

# Building a food vision using a subset of FOOD101 dataset \
    
from pathlib import Path
image_path = Path("data/pizza_steak_shushi")

# EXPLORING DATASET
import os
def walk_dir(dir_path):
    """Returns contents of dir_peth"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"there are {len(dirnames)} directories and {len(filenames)} images in {dirpath} "),
# walk_dir(image_path)
# there are 1 directories and 1 images in data\pizza_steak_shushi 
# there are 2 directories and 0 images in data\pizza_steak_shushi\pizza_steak_sushi
# there are 3 directories and 0 images in data\pizza_steak_shushi\pizza_steak_sushi\test
# there are 0 directories and 25 images in data\pizza_steak_shushi\pizza_steak_sushi\test\pizza
# there are 0 directories and 19 images in data\pizza_steak_shushi\pizza_steak_sushi\test\steak
# there are 0 directories and 31 images in data\pizza_steak_shushi\pizza_steak_sushi\test\sushi
# there are 3 directories and 0 images in data\pizza_steak_shushi\pizza_steak_sushi\train
# there are 0 directories and 78 images in data\pizza_steak_shushi\pizza_steak_sushi\train\pizza
# there are 0 directories and 75 images in data\pizza_steak_shushi\pizza_steak_sushi\train\steak
# there are 0 directories and 72 images in data\pizza_steak_shushi\pizza_steak_sushi\train\sushi

### Visualizing a RANDOM image of each type ###
from random import randrange
from PIL import Image #(PiLLOW)


# random.seed(42)
# data\pizza_steak_shushi\pizza_steak_sushi

# Get all image paths
image_path_list = list(image_path.glob("*/*/*/*.jpg")) # Goes three layers down (each star) jpg
print(image_path_list) 

# print(image_path_list) # Prints every single image path (NOt showing because space) (SHOWN AS A ARRAY)

# Pick range image path
random_path = randrange(299)
# print(random_path) # data\pizza_steak_shushi\pizza_steak_sushi\test\sushi\2394442.jpg always since random.seed()

# Access class name from path name (said in directory)
image_class = image_path_list[random_path]
image_class = image_path_list[random_path].parent.stem
# print(image_class) # sushi

# Open image using PIL (PYTHON IMAGE LIBRARY)
img = Image.open(random_path)
#Can show img.width, img.height, imge.class




    
    