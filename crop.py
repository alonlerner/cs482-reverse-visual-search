from PIL import Image
import json

# open the json file with the bounding box coordinates and images
f = open('person_boxes.json')
id = 0
data = json.load(f)
for box in data['boxes']:
    # open a image in RGB mode
    im = Image.open(box['image'])
    # set the points for cropped image
    left = box['bbox'][0]
    top = box['bbox'][1]
    right = box['bbox'][2]
    bottom = box['bbox'][3]
    # crop image of above dimension
    im1 = im.crop((left, top, right, bottom))
    # save image
    im1.save(f'./boxes_images/{id}.jpg')
    id += 1
 
f.close()