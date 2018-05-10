from PIL import Image
import numpy as np

file_name = "/Users/jeff/Documents/cs280/project/Paper_Images/ex_img_2_fore.png"

img = Image.open(file_name)

img = np.array(img)
img = np.array_split(img, [3], axis=2)[0]

ret_img = Image.fromarray(img)

ret_img.save(file_name)
