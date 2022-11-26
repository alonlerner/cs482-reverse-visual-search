#!/bin/bash
for file in ./coco_person/*.jpg
do
python3 test.py --image  ./${file}
done