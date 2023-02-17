from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("/home/shekhar/identv/train_models/runs/detect/train9/weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
results = model.predict(source="images", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("/home/shekhar/identv/train_models/images/dukantek.jpg")
results = model.predict(source=im1, save=True)  # save plotted images
im1 = Image.open("/home/shekhar/identv/train_models/images/duk2.jpg")
results = model.predict(source=im1, save=True)  # save plotted images
# im1 = Image.open("/home/shekhar/identv/train_models/images/duk3.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images