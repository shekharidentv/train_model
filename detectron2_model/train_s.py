from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import os

train_annotations_path = "./detectron2_model/dataset/annotations/instances_train.json"
train_images_path = "./dataset/SKU110K_fixed/images/"
validation_annotations_path = "./detectron2_model/dataset/annotations/instances_val.json"
validation_images_path = "./dataset/SKU110K_fixed/images/"

test_annotations_path = "./detectron2_model/dataset/annotations/instances_test.json"
test_images_path = "./dataset/SKU110K_fixed/images/"

register_coco_instances(
    "train",
    {},
    train_annotations_path,
    train_images_path
)

register_coco_instances(
    "test",
    {},
    test_annotations_path,
    test_images_path
)

register_coco_instances(
    "validation",
    {},
    validation_annotations_path,
    validation_images_path
)
metadata_train = MetadataCatalog.get("train")
dataset_dicts = DatasetCatalog.get("train")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("test",)
cfg.DATASETS.VAL = ("validation",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.OUTPUT_DIR = "./out"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()