from mtcnn import MTCNN
import cv2
import numpy as np

zoom_scale = 3/10
final_dims = (320,480)
cap = cv2.VideoCapture(0)
detector = MTCNN()
x,y,w,h = 0,0,0,0

while True:
    try:
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(640,480))
        results = detector.detect_faces(img)
        if len(results) > 0:
            if 'box' in results[0].keys():
                x,y,w,h = results[0]['box']
        crop_img = img[y-int(h*zoom_scale):y+h+int(h*zoom_scale),x-int(w*zoom_scale):x+w+int(w*zoom_scale)]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_img = cv2.resize(crop_img,final_dims)
        cv2.imshow('frame',crop_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
cap.release()
cv2.destroyAllWindows()
