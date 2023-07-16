import cv2 # OPEN CV 라이브러리를 임포트 / 이미지 및 비디오 처리 기능
import numpy as np # NumPy 라이브러리를 임포트 / 다차원 배열과 수학적 함수를 제공하여 과학적 계산에 유용
import matplotlib.pyplot as plt #  Matplotlib 라이브러리를 임포트 그래프 및 시각화 기능을 제공
import pytesseract # PyTesseract 라이브러리를 임포트 / 미지에서 텍스트를 인식하는 기능을 제공
plt.style.use('dark_background') # 그래프의 스타일을 다크모드로 설정

# 본격적인 이미지 프로세싱

img_ori = cv2.imread('1.jpg') # 파일을 cv2.imread() 함수를 사용하여 불러옴 / img_ori 변수에 이미지 데이터가 저장
height, width, channel = img_ori.shape # img_ori의 shape 속성을 사용하여 이미지의 높이, 너비, 채널 수를 추출 / height 변수에 높이, width 변수에 너비, channel 변수에 채널 수가 저장
plt.figure(figsize=(12, 10)) # 그림을 그리기 위한 새로운 Figure 객체를 생성 / figsize=(12, 10)은 그림의 크기를 가로 12인치, 세로햣 10인치로 설정
plt.imshow(img_ori, cmap='gray') # imshow() 함수를 사용하여 img_ori를 드로잉 / cmap='gray'는 그려진 이미지를 흑백으로 표시하도록 설정

<matplotlib.image.AxesImage at 0x119242f60> #  matplotlib 라이브러리에서 그려진 이미지를 나타내는 객체, 이미지를 그린 후에 해당 이미지 객체를 출력, 이미 객체의 주소가 포함된 출력

# 이미지를 그레이스케일로 변환하는 코드가 아래에 나열되요!

# hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV) # img_ori를 BGR 색상 공간에서 HSV 색상 공간으로 변환하여 hsv 변수에 저장한다. HSV 색상 공간은 색상(Hue), 채도(Saturation), 명도(Value)로 구성되며, 이미지 처리에서 색상 정보를 다루는 데 유용한 코드
# gray = hsv[:,:,2] # hsv 이미지의 채널 중 명도 채널을 추출하여 gray 변수에 저장  hsv[:,:,2]는 hsv 이미지의 세 번째 채널을 나타낸다. 
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 10)) # img_ori를 BGR 색상 공간에서 그레이스케일(흑백)로 변환하여 gray 변수에 저장한다. cv2.COLOR_BGR2GRAY는 BGR 이미지를 그레이스케일 이미지로 변환한다.
plt.imshow(gray, cmap='gray') # imshow() 함수를 사용하여 gray 이미지를 그려준다. / cmap='gray'는 그려진 이미지를 흑백으로 표시하도록 설정한다.
<matplotlib.image.AxesImage at 0x102cd2c18> # matplotlib에서 그려진 이미지를 나타내는 객체 / 이 코드는 그레이스케일 이미지를 그린 후에 해당 이미지 객체를 출력한당. 이미지 객체의 주소가 포함된 출력

# 대비 최대화(선택 사항)

structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # MORPH_RECT 형태의 구조 요소를 생성합니다. 이 구조 요소는 3x3 크기로 정의됩니다. 구조 요소는 형태학적 변환에 사용되는 커널로, 모폴로지 연산에 적용될 필터의 모양을 결정한답!
imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement) # gray 이미지에 MORPH_TOPHAT 형태의 모폴로지 연산을 적용하여 imgTopHat 이미지를 생성한다. MORPH_TOPHAT 연산은 입력 이미지와 열림(morphological opening) 연산의 차이를 계산하여 높은 주파수 성분(작은 객체나 경계선)을 강조한다.
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement) # gray 이미지에 MORPH_BLACKHAT 형태의 모폴로지 연산을 적용하여 imgBlackHat 이미지를 생성한당! MORPH_BLACKHAT 연산은 닫힘 연산과 입력 이미지의 차이를 계산하여 낮은 주파수 성분(큰 객체나 배경)을 강조한당.
imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat) # gray 이미지와 imgTopHat 이미지를 더하여 imgGrayscalePlusTopHat 이미지를 생성한다. gray 이미지에 열림 연산 결과를 더함으로써 높은 주파수 성분을 강조한다.
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat) # imgGrayscalePlusTopHat 이미지와 imgBlackHat 이미지를 뺀 결과를 gray 이미지에 저장한다. 이 연산은 낮은 주파수 성분을 강조한 후 높은 주파수 성분을 보정하는 역할을 한다.
plt.figure(figsize=(12, 10)) # 그림을 그리기 위한 새로운 Figure 객체를 생성하는 역할을 담당하는 코드이다. figsize=(12, 10)은 그림의 크기를 가로 12인치, 세로 10인치로 설정한다.
plt.imshow(gray, cmap='gray') # imshow() 함수를 사용하여 gray 이미지를 그려준다. cmap='gray'는 그려진 이미지를 흑백으로 표시하도록 설정한다.

# 적응형 임계값 지정

img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)  # gray 이미지에 5x5 크기의 가우시안 블러를 적용하여 흐림 효과를 생성한다.

img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)  # img_blurred 이미지에 적응 임계처리를 적용하여 이진화된 이미지를 생성한다. 주변 픽셀의 정보를 사용하여 임계값을 결정하고, 임계값을 넘는 픽셀은 최대값 255로 설정한다. 또,가우시안 가중치를 사용하고, 이진화된 이미지를 반전시킨다.

plt.figure(figsize=(12, 10))  # 그림을 그리기 위한 새로운 Figure 객체를 생성한다. 가로 12인치, 세로 10인치의 크기이다.

plt.imshow(img_thresh, cmap='gray')  # img_thresh 이미지를 흑백으로 시각화하여 그린다.


# 등고선 찾기

_contours, _ = cv2.findContours(
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
) # img_thresh 이미지에서 윤곽선을 찾아서 contours에 저장한다. mode=cv2.RETR_LIST는 모든 윤곽선을 찾고, method=cv2.CHAIN_APPROX_SIMPLE은 윤곽선을 간단히 표현하는 방식으로 저장한다.
temp_result = np.zeros((height, width, channel), dtype=np.uint8) # height, width, channel 크기의 영상을 생성한다. 초기 값은 모두 0으로 설정된다.
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255)) # temp_result 이미지에 찾은 모든 윤곽선을 흰색으로 그려준다.

plt.figure(figsize=(12, 10))
plt.imshow(temp_result)
# temp_result 이미지를 시각화하여 그린다.

# 날짜 준비

temp_result = np.zeros((height, width, channel), dtype=np.uint8) # height, width, channel 크기의 영상을 생성하고, 초기 값은 모두 0으로 설정된다.
contours_dict = [] # 윤곽선 정보를 저장할 빈 리스트인 contours_dict를 생성
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour) # 윤곽선을 감싸는 경계 사각형의 위치와 크기를 가져온다.

    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2) # 경계 사각형을 temp_result 이미지에 그려준다. (255, 255, 255)는 흰색이며 두께는 2로 설정된다.

    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    }) # 윤곽선 정보를 딕셔너리 형태로 contours_dict에 추가 / 윤곽선, 경계 사각형의 위치와 크기, 중심 좌표를 저장

plt.figure(figsize=(12, 10)) # 그림을 그리기 위한 새로운 Figure 객체를 생성하고, 그건 가로 12인치, 세로 10인치의 크기이다.

plt.imshow(temp_result, cmap='gray') # temp_result 이미지를 cmap='gray' 설정으로 흑백으로 그림을 그린다.

# 문자 크기로 후보 선택

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = [] # 가능한 윤곽선을 저장할 빈 리스트인 possible_contours를 생성
cnt = 0 # 윤곽선의 개수를 세기 위한 변수인 cnt를 초기화
for d in contours_dict:
    area = d['w'] * d['h'] # 윤곽선의 넓이를 계산

    ratio = d['w'] / d['h'] # 윤곽선의 가로 대비 세로 비율을 계산

    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        # 윤곽선이 최소 넓이(MIN_AREA)보다 크고, 최소 폭(MIN_WIDTH)과 최소 높이(MIN_HEIGHT)를 충족하고, 최소 비율(MIN_RATIO)과 최대 비율(MAX_RATIO) 사이에 있는 경우: 를 나타낸다.
        
        d['idx'] = cnt # 윤곽선에 cnt (인덱스)를 추가
        cnt += 1 # cnt를 1 증가
        possible_contours.append(d) # 가능한 윤곽선 리스트(possible_contours)에 해당 윤곽선 정보를 추가
        
temp_result = np.zeros((height, width, channel), dtype=np.uint8) # height, width, channel 크기의 영상 생성 / 초기 값을 모두 0으로 설정

for d in possible_contours: # 가능한 윤곽선들을 순회
    
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2) # temp_result 이미지에 해당 윤곽선을 감싸는 사각형을 드로잉 / 컬러는 흰색(255, 255, 255), 두께는 2로 설정
    
plt.figure(figsize=(12, 10)) # 그림을 그리기 위한 새로운 Figure 객체를 생성 / 가로 12인치, 세로 10인치

plt.imshow(temp_result, cmap='gray') # temp_result 이미지를 흑백으로 시각화 드로잉


# 등고선 배열로 후보 선택

MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 12.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3 # 3

def find_chars(contour_list):
    matched_result_idx = [] # 일치하는 윤곽선의 인덱스를 저장할 빈 리스트 matched_result_idx 생성

    for d1 in contour_list:
        matched_contours_idx = [] # 일치하는 윤곽선의 인덱스를 저장할 빈 리스트 matched_contours_idx 생성

        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue # 동일한 윤곽선일 경우 건너뛰기

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy']) # 윤곽선 간의 중심 좌표 차이를 계산

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) # 첫 번째 윤곽선의 대각선 길이를 계산

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])) # 윤곽선 간의 거리를 계산
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx)) # 윤곽선 간의 각도 차이를 계산

            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) # 면적
            width_diff = abs(d1['w'] - d2['w']) / d1['w'] # 폭
            height_diff = abs(d1['h'] - d2['h']) / d1['h'] # 폭 
            # 윤곽선 간의 면적, 폭, 높이 차이를 계산

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \ 
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
            # 거리, 각도, 면적, 폭, 높이의 조건을 만족하는 경우에 matched_contours_idx에 윤곽선 인덱스 추가

        # append this contour
        matched_contours_idx.append(d1['idx']) # 현재 윤곽선의 인덱스를 matched_contours_idx에 추가

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue # 일치하는 윤곽선의 개수가 최소 개수(MIN_N_MATCHED)보다 작은 경우 건너뛰기

        matched_result_idx.append(matched_contours_idx) # 일치하는 윤곽선의 인덱스 리스트를 matched_result_idx에 추가

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])
        # 일치하지 않는 윤곽선의 인덱스를 unmatched_contour_idx에 추가

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx) # 일치하지 않는 윤곽선을 possible_contours에서 추출

        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        # 재귀적으로 일치하는 윤곽선을 찾기

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
        # 재귀적으로 찾은 일치하는 윤곽선의 인덱스를 matched_result_idx에 추가

        break

    return matched_result_idx # 일치하는 윤곽선의 인덱스 리스트를 반환

result_idx = find_chars(possible_contours) # find_chars 함수를 호출하여 가능한 윤곽선들 중 일치하는 윤곽선의 인덱스 리스트를 얻음

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))
# 일치하는 윤곽선들의 정보를 matched_result에 추가

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
# height, width, channel 크기의 영상을 생성 / 초기 값은 모두 0

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        # matched_result에 있는 윤곽선들을 temp_result 이미지에 감싸는 사각형으로 그린다.

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
# temp_result 이미지를 흑백으로 시각화하여 드로잉

# 플레이트 이미지 회전

MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 12.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3 # 3

def find_chars(contour_list):
    matched_result_idx = []
    # 일치하는 윤곽선의 인덱스를 저장할 빈 리스트인 matched_result_idx를 생성

    for d1 in contour_list:
        matched_contours_idx = []
        # 일치하는 윤곽선의 인덱스를 저장할 빈 리스트인 matched_contours_idx를 생성

        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            # 동일한 윤곽선일 경우 건너뛰기

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            # 윤곽선 간의 중심 좌표 차이를 계산

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            # 첫 번째 윤곽선의 대각선 길이를 계산

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            # 윤곽선 간의 거리를 계산
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            # 윤곽선 간의 각도 차이를 계산

            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']
            # 윤곽선 간의 면적, 폭, 높이 차이를 계산

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
            # 거리, 각도, 면적, 폭, 높이의 조건을 만족하는 경우 matched_contours_idx에 윤곽선 인덱스 추가

        # append this contour
        matched_contours_idx.append(d1['idx'])
        # 현재 윤곽선의 인덱스도 matched_contours_idx에 추가

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue
        # 일치하는 윤곽선의 개수가 최소 개수(MIN_N_MATCHED)보다 작은 경우 건너뛰기

        matched_result_idx.append(matched_contours_idx)
        # 일치하는 윤곽선의 인덱스 리스트를 matched_result_idx에 추가

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])
        # 일치하지 않는 윤곽선의 인덱스를 unmatched_contour_idx에 추가

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        # 일치하지 않는 윤곽선을 possible_contours에서 추출

        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        # 재귀적으로 일치하는 윤곽선을 찾음

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
        # 재귀적으로 찾은 일치하는 윤곽선의 인덱스를 matched_result_idx에 추가

        break

    return matched_result_idx
    # 일치하는 윤곽선의 인덱스 리스트를 반환

result_idx = find_chars(possible_contours)
# find_chars 함수를 호출하여 가능한 윤곽선들 중 일치하는 윤곽선의 인덱스 리스트를 얻음

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))
# 일치하는 윤곽선들의 정보를 matched_result에 추가

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
# height, width, channel 크기의 영상을 생성 / 초기 값은 모두 0

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        # matched_result에 있는 윤곽선들을 temp_result 이미지에 감싸는 사각형으로 드로잉

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
# temp_result 이미지를 흑백으로 드로잉


    # 문자를 찾기 위한 또 다른 임계값

    longest_idx, longest_text = -1, 0
plate_chars = []
# 각 차량 번호판 이미지에 대해 반복
for i, plate_img in enumerate(plate_imgs): # 번호판 이미지를 크기 조정
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6) # 번호판 이미지를 이진화
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 윤곽선을 다시 찾기. (위와 동일한 과정)
    _, contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)  # 번호판 이미지의 최소 좌표와 최대 좌표를 초기화
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0 
    # 윤곽선을 순회하며 번호판 이미지의 최소 좌표와 최대 좌표를 갱신
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
    # 번호판 영역만 추출
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x] # 추출된 번호판 영역을 가우시안 블러로 블러링
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0) # 번호판 영역을 다시 이진화
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 번호판 영역 주위에 경계 추가
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0)) # 번호판 영역에서 문자를 추출
    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
    result_chars = ''
    has_digit = False
    # 추출된 문자 중에서 한글과 숫자만 선택
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print(result_chars)
    plate_chars.append(result_chars)
    # 숫자가 있고, 추출된 문자열의 길이가 가장 긴 경우를 선택
    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

    plt.subplot(len(plate_imgs), 1, i+1)
    plt.imshow(img_result, cmap='gray')

# 가장 긴 문자열을 가진 번호판 정보를 선택
info = plate_infos[longest_idx]
chars = plate_chars[longest_idx]

print(chars)

img_out = img_ori.copy()

# 가장 긴 문자열을 가진 번호판 영역을 사각형으로 표시
cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

# 결과 이미지를 저장
cv2.imwrite(chars + '.jpg', img_out)

plt.figure(figsize=(12, 10))
plt.imshow(img_out)
