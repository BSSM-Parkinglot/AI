import cv2
import pytesseract
from pytesseract import Output
import re
import mysql.connector

# 경로 설정
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


def preprocess_and_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)

    return thresh


def is_valid_license_plate(license_plate_parts):
    pattern = r"^\d{2}\w{1}\d{4}$"
    combined_text = "".join(license_plate_parts)
    return bool(re.match(pattern, combined_text))


def save_to_database(text):
    # 데이터베이스와 연결 설정
    cnx = mysql.connector.connect(user='root', password='mysql', host='svc.sel4.cloudtype.app', database='parkinglot', port='32676')
    cursor = cnx.cursor()

    # 텍스트를 데이터베이스에 저장
    query = "UPDATE Spot_Info SET Parking = 1, ParkingNum = %s WHERE Spot = %s"
    cursor.execute(query, (text, 'A1'))

    # 변동 사항 커밋
    cnx.commit()

    # 연결 해제
    cursor.close()
    cnx.close()

def open_Breaker_to_database():
    # 데이터베이스와 연결 설정
    cnx = mysql.connector.connect(user='root', password='mysql', host='svc.sel4.cloudtype.app', database='parkinglot', port='32676')
    cursor = cnx.cursor()

    # 텍스트를 데이터베이스에 저장
    query = "UPDATE Breaker SET Situation = 1 WHERE `Exit` = 1"
    cursor.execute(query, ())

    # 변동 사항 커밋
    cnx.commit()

    # 연결 해제
    cursor.close()
    cnx.close()

def close_Breaker_to_database():
    # 데이터베이스와 연결 설정
    cnx = mysql.connector.connect(user='root', password='mysql', host='svc.sel4.cloudtype.app', database='parkinglot', port='32676')
    cursor = cnx.cursor()

    # 텍스트를 데이터베이스에 저장
    query = "UPDATE Breaker SET Situation = 0 WHERE `Exit` = 1"
    cursor.execute(query, ())

    # 변동 사항 커밋
    cnx.commit()

    # 연결 해제
    cursor.close()
    cnx.close()


def find_combinations(recognized_texts):
    combined_text = ' '.join(recognized_texts)
    pattern = r'\d{2}\s?[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]\s?\d{4}'
    matches = re.findall(pattern, combined_text)
    return [match.replace(' ', '') for match in matches]

cnt = 0
close = 0

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
        filtered_text = ''.join(c for c in text if c.isalnum() and c not in ('"', "'", "*", "&", "^", "$", "~", "!", "₩", " ", ".", "ㅇ", "ㅠ", "ㆍ", "_", "ㅡ"))
        if len(filtered_text) > 0:
            recognized_texts.append(filtered_text)

    license_plate_combinations = find_combinations(recognized_texts)

    if license_plate_combinations:
        for combination in license_plate_combinations:
            if is_valid_license_plate([combination]):
                close = 0
                print(f"---- {combination} ----")
                print("출입구가 열렸습니다.")
                open_Breaker_to_database()
                print('자동차 정보가 저장되었습니다.')
    elif len(recognized_texts) > 0:
        print(recognized_texts)
        close = 0
        cnt += 1
    else:
        close += 1
        if (60 == close):
            print('출입구가 닫혔습니다.')
            close_Breaker_to_database()
            close = 0

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
