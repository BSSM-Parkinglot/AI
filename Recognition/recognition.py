import cv2
import pytesseract
from pytesseract import Output
import re

# 경로 설정
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


# 이미지 전처리 및 텍스트 영역 검출 함수
def preprocess_and_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)

    return thresh


# 올바른 번호판 형식인지 확인하는 함수
def is_valid_license_plate(text):
    pattern = r"^\d{2}\w{1}\d{4}$"
    return bool(re.match(pattern, text))


# 웹캠을 사용한 실시간 처리
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    preprocessed = preprocess_and_detect(frame)
    data = pytesseract.image_to_data(preprocessed, lang='kor', output_type=Output.DICT)

    license_plate_parts = []  # 번호판 부분 저장 리스트 초기화
    for i in range(len(data['level'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data['text'][i]

        if len(text) > 0:
            license_plate_parts.append(text)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        combined_text = "".join(license_plate_parts)
        if is_valid_license_plate(combined_text):
            print(combined_text)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
