from PIL import Image
import json

f = open('person_boxes.json')
id = 0
data = json.load(f)
for box in data['boxes']:
    # Opens a image in RGB mode
    im = Image.open(box['image'])

    # Setting the points for cropped image
    left = box['bbox'][0]
    top = box['bbox'][1]
    right = box['bbox'][2]
    bottom = box['bbox'][3]
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1.save(f'./boxes_images/{id}.jpg')
    id += 1
 
# Shows the image in image viewer
f.close()