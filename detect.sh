#!/bin/bash
for each file in /coco_person/*.jpg
do
python test.py --image  /coco_person/${file}
end