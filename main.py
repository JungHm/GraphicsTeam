# https://devhyeon.tistory.com/19
import random # 가위바위보에 사용할 랜덤
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image



# 가위바위보 변수
sel = ['가위', '바위', '보']
result = {0: '승리했습니다.', 1: '패배했습니다.', 2: '비겼습니다.'}
start = 0
# 가위바위보 체크
def checkWin(user, com): # 0:Draw 1:Win -1:Lose
    print(f'사용자 ( {user} vs {com} ) 컴퓨터')
    if user == com:
        return 0
    elif user == '가위' and com == '바위':
        return -1
    elif user == '바위' and com == '보':
        return -1
    elif user == '보' and com == '가위':
        return -1
    else:
        return 1

fontpath = "NanumGothic.ttf"
#모델 로드
model = load_model('RPS.h5')
model.summary()
 
# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()
      
# loop through frames
while webcam.isOpened():
    # read frame from webcam 
    status, frame = webcam.read()
    
    if not status:
        break
    img = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0]) # 예측된 클래스 0, 1, 2
    print(prediction[0])
    print(predicted_class)
    
    if predicted_class == 0:
        me = "바위"
    elif predicted_class == 1:
        me = "보"        
    elif predicted_class == 2:
        me = "가위"

    # # 특정 키 눌렀을 때 바로 동작되게?
    if cv2.waitKey(1) & 0xFF == ord('g'):
        user = me # input 가위, 바위, 보 
        com = sel[random.randint(0, 2)]
        check = checkWin(user, com) == 0
        if check == 1:
            me = "Win"
        elif check == 0:
            me = "Draw"
        elif check == -1:
            me = "Lose"
        # 이 부분에서 결과 출력 & pause 2~3초간 
        font1 = ImageFont.truetype(fontpath, 100)
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
        frame = np.array(frame_pil)
        cv2.imshow('RPS', frame)
        start = time.time()
        continue
        

                
    # display
    if time.time() - start > 2:
        font1 = ImageFont.truetype(fontpath, 100)
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
        frame = np.array(frame_pil)
        cv2.imshow('RPS', frame)
        
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()