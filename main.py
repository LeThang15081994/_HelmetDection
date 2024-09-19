from ultralytics import YOLOv10
import torch
import matplotlib.pyplot as plt
import cv2

def checks_torch():
    if torch.cuda.is_available(): 
        print('it Work')

def train_model(path_model, path_yaml, img_size, batch_size, devices, epochs_size): 
    
    #Load model
    model = YOLOv10(path_model)
    
    #model.info()
    epochs_size = 25
    #train
    model.train(data=path_yaml,
                epochs = epochs_size,
                imgsz = img_size,
                batch = batch_size,
                device = devices)  # device ='cuda' to choose GPU
  
def val_model(path_model, path_yaml, img_size, batch_size, devices): 
    
    #Load model
    model = YOLOv10(path_model)

    model.val(data=path_yaml,
                batch = batch_size,
                imgsz = img_size,
                device = devices,
                split='test')

def predic_model(path_model, img_size, confiden, path_predict): 
    
    #config
    model = YOLOv10(path_model)
    #predic
    results= model.predict(source = path_predict, 
                            imgsz = img_size,
                            save = True, 
                            conf = confiden)
    img = results[0].plot()
    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Predicted Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   
if __name__ == '__main__':
   
   #config
   path_model = './yolov10/runs/detect/train/weights/best.pt'
   path_yaml = './safety-Helmet-Reflective-Jacket/data_helmet.yaml'
   path_predict = './testImg.jpg'
   epochs_size = 25
   batch_size = 16
   img_size = 640
   devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   confiden = 0.4

   #checks_torch()
   #train_model('yolov10s.pt', path_yaml, img_size, batch_size, devices, epochs_size)
   val_model(path_model, path_yaml, img_size, batch_size, devices)
   predic_model(path_model, img_size,confiden, path_predict)