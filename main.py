from ultralytics import YOLOv10
import torch

def checks_torch():
    if torch.cuda.is_available(): 
        print('it Work')

def train_model(): 
    
    #config
    model = YOLOv10('yolov10s.pt')

    #model.info()
    epochs_size = 50
    img_size = 640
    batch_size = 16
    path_yaml = "../safety-Helmet-Reflective-Jacket/data.yaml"

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
    path_yaml = "../safety-Helmet-Reflective-Jacket/data.yaml"

    #Val
    model.val(data=path_yaml,
                batch = batch_size)

def predic_model(): 
    
    #config
    model = YOLOv10('./yolov10/runs/detect/train/weights/best.pt')

    batch_size = 16
    path_yaml = "../safety-Helmet-Reflective-Jacket/test/"

    #Val
    model.predict(source = "./safety-Helmet-Reflective-Jacket/test/images/", save = True, conf = 0.5)

   
if __name__ == '__main__':
   #checks_torch()
   #train_model()
   #val_model()
   #predic_model()