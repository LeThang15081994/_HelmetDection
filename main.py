from ultralytics import YOLOv10
import torch
import matplotlib.pyplot as plt
import cv2

def checks_torch():
    if torch.cuda.is_available(): 
        print('it Work')

def train_model(): 
    
    #config
    model = YOLOv10('yolov10s.pt')

    #model.info()
    epochs_size = 25
    img_size = 640
    batch_size = 16
    path_yaml = 'D:/WorkSpace/_thangle15894/_AIproject/_HelmetDetection/safety-Helmet-Reflective-Jacket/data_helmet.yaml'

    #train
    model.train(data=path_yaml,
                epochs = epochs_size,
                imgsz = img_size,
                batch = batch_size,
                device = 'cuda')
    # device ='cuda' to choose GPU

def val_model(): 
    
    #config
    model = YOLOv10('./yolov10/runs/detect/train/weights/best.pt')

    batch_size = 16
    path_yaml = "./safety-Helmet-Reflective-Jacket/data_helmet.yaml"
    img_size = 640

    #Val
    model.val(data=path_yaml,
                batch = batch_size,
                imgsz = img_size,
                device = 'cuda',
                split='test')

def predic_model(): 
    
    #config
    model = YOLOv10('./yolov10/runs/detect/train/weights/best.pt')
    img_size = 640
    #predic
    results= model.predict(source = "./testImg.jpg", 
                  imgsz = img_size,
                  save = True, 
                  conf = 0.40)
    print(len(results))

   
if __name__ == '__main__':
   #checks_torch()
   #train_model()
   #val_model()
   predic_model()