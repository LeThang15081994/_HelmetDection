from ultralytics import YOLOv10
import torch
import matplotlib.pyplot as plt
import cv2

class HelmetDetection:
    def __init__(self, path_model, path_yaml, img_size=640, batch_size=16, epochs_size=25):
        self.path_model = path_model
        self.path_yaml = path_yaml
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs_size = epochs_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def checks_torch(self):
        if torch.cuda.is_available(): 
            print('It works with GPU.')
        else:
            print('Using CPU.')

    def train_model(self, path_pretrain):
        self.model = YOLOv10(path_pretrain) # đường dẫn tới pretrain yolov10s.pt
        self.model.train(data=self.path_yaml,
                         epochs=self.epochs_size,
                         imgsz=self.img_size,
                         batch=self.batch_size,
                         device=self.device)

    def val_model(self):
        self.model = YOLOv10(self.path_model) # đường dẫn tới đã huấn luyện
        self.model.val(data=self.path_yaml,
                       batch=self.batch_size,
                       imgsz=self.img_size,
                       device=self.device,
                       split='test')

    def predict_model(self, path_predict, confiden):
        self.model = YOLOv10(self.path_model) # đường dẫn tới model đã huấn huyện
        results = self.model.predict(source=path_predict, 
                                      imgsz=self.img_size,
                                      save=True, 
                                      conf=confiden)
        img = results[0].plot()
        cv2.imshow("Predicted Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Config
    path_pretrain = './yolov10s.pt'
    path_model = './yolov10/runs/detect/train/weights/best.pt'
    path_yaml = './safety-Helmet-Reflective-Jacket/data_helmet.yaml'
    path_predict = './testImg.jpg'
    
    # Các giá trị mặc định
    img_size = 640
    batch_size = 16
    epochs_size = 25
    confiden = 0.5

    helmet_detection = HelmetDetection(path_model, path_yaml, img_size, batch_size, epochs_size)
    helmet_detection.checks_torch()
    helmet_detection.train_model(path_pretrain)
    helmet_detection.val_model()
    helmet_detection.predict_model(path_predict, confiden)






'''def checks_torch():
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
   predic_model(path_model, img_size,confiden, path_predict)'''