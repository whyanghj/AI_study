import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import time

import os
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MEDIAPIPE_DISABLE_TFLITE_DELEGATE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_x, image_y = 200, 200
num_of_classes = 7

# 코드에서 사용하는 클래스 매핑
chord_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

# CNN 모델 정의
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 19 * 19, 1024)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 모델 초기화 및 가중치 로드
model = CNNModel(num_of_classes).to(device)
model.load_state_dict(torch.load('guitar_learner.pth'))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((image_x, image_y)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# OpenCV 영상 캡처 설정
cap = cv2.VideoCapture(0)

# 메시지 출력 시간 제어 변수
last_no_hand_time = time.time()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def main():
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)

    global last_no_hand_time

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image_rgb)

        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)

            # 손 영역 추출 및 예측
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea)
                contour = contours[-1]
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = gray[y1:y1 + h1, x1:x1 + w1]
                save_img = cv2.resize(save_img, (image_x, image_y))

                # PyTorch 모델로 예측
                pred_probab, pred_class = pytorch_predict(model, save_img)
                if pred_class != -1:
                    print(f"Predicted class: {chord_dict[pred_class]}, Probability: {pred_probab:.2f}")
                    cv2.putText(image, str(chord_dict[pred_class]), (x1 + 50, y1 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                else:
                    print("No confident prediction made.")
        else:
            # 손이 인식되지 않았을 때, 1초 간격으로만 메시지 출력
            current_time = time.time()
            if current_time - last_no_hand_time > 1:  # 1초 간격
                print("No hand detected.")
                last_no_hand_time = current_time

        # 화면 표시 및 종료 조건
        image = rescale_frame(image, percent=75)
        cv2.imshow("Img", image)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

def pytorch_predict(model, image):
    # 이미지 전처리 및 예측
    if image.sum() == 0:  # 이미지가 완전히 검정색인 경우 예측하지 않음
        return 0.0, -1

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        pred_probab, pred_class = torch.max(probabilities, dim=1)

        # 확률이 낮으면 예측하지 않음
        if pred_probab.item() < 0.6:
            return 0.0, -1

        return pred_probab.item(), pred_class.item()

main()
