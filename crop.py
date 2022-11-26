from PIL import Image
 
# Opens a image in RGB mode
im = Image.open(r"./coco_person/000000000036.jpg")
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
 
# Setting the points for cropped image
left = 168
top = 151
right = 449
bottom = 639
 
# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))
 
# Shows the image in image viewer
im1.show()