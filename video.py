import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import cv2
import datetime
import PIL
from model import Net_1, Net_2



def preprocess(img):
  # img=PIL.Image.fromarray(img)
  data_transforms=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
  img = data_transforms(img)
  img = img.float()
  img = img.unsqueeze(0)
  return img      

def argmax(prediction):
  prediction = prediction.cpu()
  prediction = prediction.detach().numpy()
  print("pred shape: ",prediction.shape)
  # print("prediction ",prediction)
  top_1 = np.argmax(prediction, axis=1)
  score = np.amax(prediction)
  prediction = top_1[0]
  #This is the Label
  Labels = { 0 : 'A',
          1 : 'B',
          2 : 'C',
          3 : 'D',
          4 : 'E',
          5 : 'F',
          6 : 'G',
          7 : 'H',
          8 : 'I',
          9:  'J',
          10 : 'K',
          11: 'L',
          12: 'M',
          13: 'N',
          14: 'O',
          15: 'P',
          16: 'Q',
          17: 'R',
          18: 'S',
          19: 'T',
          20: 'U',
          21: 'V',
          22: 'W',
          23: 'X',
          24: 'Y',
          25: 'Z',
          26: 'del',
          27: 'nothing',
          28: 'space'
      }
  result = Labels[prediction]

  return result,score

def main():
  model=Net_1()
  model.load_state_dict(torch.load("trained_model_torch_exp1.pth"))
  
  
  fps=0
  text_editor=""

  cap=cv2.VideoCapture(0)

  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
    
  size = (frame_width, frame_height)
    
  video_w = cv2.VideoWriter('India.avi', 
                          cv2.VideoWriter_fourcc(*'MJPG'),20,size)
  
  while cap.isOpened():
    with torch.no_grad():
      model.eval()
      _,frame=cap.read()
      if fps==60:
        fps=0
        image=frame.copy()
        image1=image[10:210,10:210]
        image = PIL.Image.fromarray(image)
        
        image_data = preprocess(image1)
      
        prediction   = model(image_data)
        result,score = argmax(prediction)
        
        if result=="space":
          text_editor+=" "
        elif result=="del":
          text_editor=text_editor[:-1]
        elif result=="nothing":
          text_editor+=""
        else:
          text_editor+=result
    cv2.putText(frame, '%s' %(text_editor),(450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    cv2.rectangle(frame,(10,10),(210,210), (250,0,0), 2)
    video_w.write(frame)
    cv2.imshow("win1",frame)
    key = cv2.waitKey(1)
    if key == 27:
      break
    fps+=1
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()