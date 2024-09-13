from ultralytics import YOLOv10

model = YOLOv10('yolov10s.pt')
#model.info()

epochs_size = 5
img_size = 640
batch_size = 254
path_yaml = "../safety-Helmet-Reflective-Jacket/data.yaml"

model.train(data=path_yaml,
            epochs = epochs_size,
            imgsz = img_size,
            batch = batch_size)