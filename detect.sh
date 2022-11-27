#!/bin/bash
for file in ./coco_person/*.jpg
do
python3 detect-person.py --image  ./${file}
done