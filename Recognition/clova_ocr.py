import cv2
import requests
import re
import mysql.connector

# 네이버 CLOVA 인증 정보
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
naver_ocr_url = "https://naveropenapi.apigw.ntruss.com/recog/v1/ocr"

headers = {
    "X-NCP-APIGW-API-KEY-ID": client_id,
    "X-NCP-APIGW-API-KEY": client_secret
}


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

def delete_to_database():
    # 데이터베이스와 연결 설정
    cnx = mysql.connector.connect(user='root', password='mysql', host='svc.sel4.cloudtype.app', database='parkinglot', port='32676')
    cursor = cnx.cursor()

    # 텍스트를 데이터베이스에 저장
    query = "UPDATE Spot_Info SET Parking = 0, ParkingNum = NULL WHERE Spot = %s"
    cursor.execute(query, ('A1',))

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

def ocr_naver(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {"image": ("frame.jpg", img_encoded, "image/jpg")}
    response = requests.post(naver_ocr_url, headers=headers, files=files)
    result = response.json()

    recognized_texts = []
    if "recognition" in result and "items" in result["recognition"]:
        for item in result["recognition"]["items"]:
            text = item["text"].strip()
            filtered_text = ''.join(c for c in text if c.isalnum() and c not in ('"', "'", "*", "&", "^", "$", "~", "!", "₩", " ", ".", "ㅇ", "ㅠ", "ㆍ", "_", "ㅡ"))
            if len(filtered_text) > 0:
                recognized_texts.append(filtered_text)

    return recognized_texts


cnt = 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    preprocessed = preprocess_and_detect(frame)
    recognized_texts = ocr_naver(preprocessed)
    license_plate_combinations = find_combinations(recognized_texts)

    if license_plate_combinations:
        for combination in license_plate_combinations:
            if is_valid_license_plate([combination]):
                print(f"---- {combination} ----")
                save_to_database(combination)

    elif len(recognized_texts) > 0:
        print(recognized_texts)
        cnt += 1
    elif cnt == 15:
        print('삭제됨')
        delete_to_database()
        cnt = 0

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
