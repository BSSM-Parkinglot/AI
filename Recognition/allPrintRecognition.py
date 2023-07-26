import cv2
import pytesseract
from pytesseract import Output

# 경로 설정
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


def preprocess_and_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)

    return thresh


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    preprocessed = preprocess_and_detect(frame)
    data = pytesseract.image_to_data(preprocessed, lang='kor', output_type=Output.DICT)

    recognized_texts = []
    for i in range(len(data['level'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, data['text'][i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        text = data['text'][i]
        filtered_text = ''.join(c for c in text if c.isalnum() and c not in ('"', "'", "*", "&", "^", "$", "~", "!", "₩", " ", ".", "ㅇ", "ㅠ", "ㆍ"))
        if len(filtered_text) > 0:
            recognized_texts.append(filtered_text)

    if len(recognized_texts) > 0:
        print(recognized_texts)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
