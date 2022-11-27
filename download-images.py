from pycocotools.coco import COCO
import requests

# use ms coco mini
ann_file = "./instances_minitrain2017.json"
coco=COCO(ann_file)

# get all person images
imgIds = coco.getImgIds(catIds=[1])
images = coco.loadImgs(imgIds)

# download the person images
for im in images:
    img_data = requests.get(im['coco_url']).content
    with open('./coco_person/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)