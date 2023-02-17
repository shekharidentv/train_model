import csv
import datetime
import json
import os

# The CSV columns are: image_name,x1,y1,x2,y2,class,image_width,image_height

class data_mapper():

    def __init__(self):
        self.images = {}
        self.image_count = 0
        self.categories = {}
        self.category_index = 0
        self.dataset = {}
        self.image_metadata = {}

    def get_image_id(self, image_name):
        check = self.images.get(image_name, None)
        if not check:
            self.image_count += 1
            self.images[image_name] = self.image_count
            return self.image_count
        else:
            return check

    def get_categories_id(self, categories_name):
        check = self.categories.get(categories_name, None)
        if not check:
            self.category_index += 1
            self.categories[categories_name] = self.category_index
            return self.category_index
        else:
            return check

    def add_categories(self):
        self.dataset["categories"] = [
            {
                "supercategory": cat_name,
                "id": cat_index,
                "name": cat_name,
            } for cat_name, cat_index in self.categories.items()]

    def add_images(self):
        for image_name, image_meta in self.image_metadata.items():
            now = datetime.datetime.now()
            self.dataset["images"].append(
                {
                    "license": 1,
                    "file_name": f"./dataset/SKU110K_fixed/images/{image_name}",
                    "coco_url": f"./dataset/SKU110K_fixed/images/{image_name}",
                    "height": int(image_meta["height"]),
                    "width": int(image_meta["width"]),
                    "date_captured": now.strftime("%m/%d/%Y"),
                    "id": image_meta["id"],
                }
            )

    def process(self, csv_path='./dataset/SKU110K_fixed/annotations/annotations_test.csv'):
        self.images = {}
        self.image_count = 0
        self.categories = {}
        self.category_index = 0
        self.dataset = {}
        self.image_metadata = {}
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            now = datetime.datetime.now()
            self.dataset = {
                "info": {
                    "description": f"retail project",
                    "url": "http://cocodataset.org",
                    "version": "1.0",
                    "year": "2022",
                    "contributor": "shekhar koirala",
                    "date_created": now.strftime("%m/%d/%Y"),
                },
                "licenses": [
                    {
                        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                        "id": 1,
                        "name": "Attribution-NonCommercial-ShareAlike License",
                    }
                ],
                "images": [],
                "annotations": [],
                "categories": [],
                "segment_info": [],
            }
            bbox_count = 0
            for row in csv_reader:
                image_id = self.get_image_id(row[0])
                self.dataset["annotations"].append({
                        "segmentation": [],
                        "area": (int(row[3]) - int(row[1]))
                                * (int(row[4]) - int(row[2])),
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [
                            int(row[1]),
                            int(row[2]),
                            int(row[3]) - int(row[1]),
                            int(row[4]) - int(row[2]),
                        ],
                        "category_id": self.get_categories_id(row[5]),
                        "id": bbox_count,
                    }
                )
                self.image_metadata[row[0]] = {"width": row[6], "height": row[7], "id": image_id}
                bbox_count += 1
            self.add_images()
            self.add_categories()
        return self.dataset


path_dir = f"dataset/SKU110K_fixed"
dm = data_mapper()
test_data = dm.process(f'{path_dir}/annotations/annotations_test.csv')
val_data = dm.process(f'{path_dir}/annotations/annotations_val.csv')
train_data = dm.process(f'{path_dir}/annotations/annotations_train.csv')



coco_output_dir = "./detectron2_model/dataset/annotations"
if not os.path.exists(coco_output_dir):
    os.makedirs(coco_output_dir)
with open(f'{coco_output_dir}/instances_test.json', 'w') as fp:
    json.dump(test_data, fp)
with open(f'{coco_output_dir}/instances_val.json', 'w') as fp:
    json.dump(val_data, fp)
with open(f'{coco_output_dir}/instances_train.json', 'w') as fp:
    json.dump(train_data, fp)