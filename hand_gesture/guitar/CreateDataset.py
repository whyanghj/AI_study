import cv2
import os
import mediapipe as mp

# MediaPipe의 손 인식 모듈과 랜드마크 그리기 도구를 가져옴
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 이미지 크기 설정
image_x, image_y = 200, 200

# 폴더를 생성하는 함수
def create_folder(folder_name):
    """
    지정된 폴더가 존재하지 않으면 생성
    :param folder_name: 생성할 폴더의 이름
    """
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

# 주요 기능 구현 함수
def main(c_id):
    """
    실시간으로 웹캠을 통해 손을 인식하고 이미지를 캡처하여 저장
    :param c_id: 저장할 코드(예: 기타 코드) 폴더 이름
    """
    # 캡처할 총 이미지 수 설정
    total_pics = 1200

    # MediaPipe 손 인식 초기화
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,  # 손 인식 최소 신뢰도
        min_tracking_confidence=0.5   # 손 추적 최소 신뢰도
    )

    # 랜드마크와 연결선의 스타일 설정
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=5)  # 빨간색 랜드마크
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=10)  # 선 두께 2

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    # 저장 폴더 생성
    create_folder("chords/" + str(c_id))

    # 이미지 캡처 관련 변수 초기화
    pic_no = 0
    flag_start_capturing = False  # 캡처 시작 여부
    frames = 0  # 캡처된 프레임 수

    while cap.isOpened():
        # 웹캠에서 이미지 읽기
        ret, image = cap.read()
        image = cv2.flip(image, 1)  # 좌우 반전
        image_orig = cv2.flip(image, 1)  # 원본 이미지를 유지
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환

        # MediaPipe로 손 인식 수행
        results_hand = hands.process(image)

        # 이미지 쓰기 가능 상태로 설정
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 다시 BGR로 변환

        # 손 랜드마크가 감지되었을 때
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                # 랜드마크와 연결선 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec
                )

                # 랜드마크 좌표 추출
                h, w, _ = image.shape
                landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # 모든 랜드마크의 최소/최대 좌표 계산
                x_coords = [p[0] for p in landmark_points]
                y_coords = [p[1] for p in landmark_points]

                # 확장된 경계 상자 좌표 계산
                x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)  # 최소 좌표에 여유 공간 추가
                x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)  # 최대 좌표에 여유 공간 추가

                # 경계 상자 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 경계 상자 내부를 추출하고 리사이즈
                save_img = gray[y1:y2, x1:x2]
                save_img = cv2.resize(save_img, (image_x, image_y))

                # 이미지 저장
                pic_no += 1
                cv2.imwrite("chords/" + str(c_id) + "/" + str(pic_no) + ".jpg", save_img)

                # 캡처 상태와 번호 표시
                cv2.putText(image, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.putText(image, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))

                # 설정된 이미지 수에 도달하면 종료
                if pic_no == total_pics:
                    break

        # 결과 이미지와 추출 이미지를 화면에 출력
        cv2.imshow("Capturing gesture", image)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            break

    # 손 인식 종료 및 웹캠 해제
    hands.close()
    cap.release()

# 코드 실행
c_id = input("Enter chord: ")  # 저장할 코드(폴더 이름) 입력
main(c_id)  # main 함수 실행
