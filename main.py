from ultralytics import YOLOv10
import torch

def checks_torch():
    if torch.cuda.is_available(): 
        print('it Work')

def train_model(): 
    
    #config
    model = YOLOv10('yolov10s.pt')

    model.info()
    epochs_size = 100
    img_size = 640
    batch_size = 16
    path_yaml = "../safety-Helmet-Reflective-Jacket/data.yaml"

    #train
    model.train(data=path_yaml,
                epochs = epochs_size,
                imgsz = img_size,
                batch = batch_size,
                device = 'cuda')
    # device ='cuda' to choose GPU. if have muti GPU choose "0", "1"
   
if __name__ == '__main__':
   #checks_torch()
   train_model()
   