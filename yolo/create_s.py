import os
import sys
import shutil

import csv
import datetime
import json

# The CSV columns are: image_name,x1,y1,x2,y2,class,image_width,image_height
# yolo format columns are:


class data_mapper():
    def __init__(self, src_path):
        self.src_path = src_path

    def convert_to_yolo(self, bboxes, image_width, image_height):
        x1, y1, x2, y2 = bboxes
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        return [str(x) for x in [0, x_center, y_center, width, height]]

    def process(self, csv_path='./dataset/SKU110K_fixed/annotations/annotations_test.csv', dst_dir = None):
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            now = datetime.datetime.now()
            selected_image = -1
            old_images = []
            for row in csv_reader:
                if selected_image != row[0] and old_images:
                    # do process here.
                    name = selected_image.split(".")[0]
                    shutil.copy(f"{self.src_path}/{selected_image}", f"{dst_dir}/images")
                    print(f"{self.src_path}/{selected_image}", f"{dst_dir}/images", row[0], selected_image, len(old_images))
                    with open(f'{dst_dir}/labels/{name}.txt', 'w') as f:
                        for sublist in old_images:
                            yolo = self.convert_to_yolo([int(x) for x in [sublist[1], sublist[2], sublist[3], sublist[4]]],
                                                        int(sublist[6]), int(sublist[7]))
                            row = "\t".join(yolo) + "\n"
                            f.write(row)
                    line_count += 1
                    # if not line_count % 100:
                    #     break
                    old_images = []
                else:
                    old_images.append(row)
                    selected_image = row[0]

print(os.getcwd())
sys.path.append("/home/shekhar/identv/train_models")

path_dir = f"./dataset/SKU110K_fixed"
main_path = "./yolo/dataset"
for mode_ in ["train", "test", "validation"]:
    for data_type in ["images", "labels"]:
        specific_path = f"{main_path}/{mode_}/{data_type}"
        if not os.path.exists(specific_path):
            os.makedirs(specific_path)

dm = data_mapper(src_path="./dataset/SKU110K_fixed/images")
dm.process(f'{path_dir}/annotations/annotations_test.csv', dst_dir="./yolo/dataset/test")
dm.process(f'{path_dir}/annotations/annotations_val.csv', dst_dir="./yolo/dataset/validation")
dm.process(f'{path_dir}/annotations/annotations_train.csv', dst_dir="./yolo/dataset/train")
#
#
#

# with open(f'{coco_output_dir}/instances_test.json', 'w') as fp:
#     json.dump(test_data, fp)
# with open(f'{coco_output_dir}/instances_val.json', 'w') as fp:
#     json.dump(val_data, fp)
# with open(f'{coco_output_dir}/instances_train.json', 'w') as fp:
#     json.dump(train_data, fp)