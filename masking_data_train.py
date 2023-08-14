# 해당 폴더는 data 중 train을 masking 하는 과정입니다

import cv2
import json
import numpy as np
import os

# 이미지 크기 정보
image_width = 1920
image_height = 1080

# 디렉토리 경로
json_dir = "C:\\Users\\gyqls\\Teamproject2\\masking_images\\data\\labels\\train"
image_dir = "C:\\Users\\gyqls\\Teamproject2\\masking_images\\data\\images\\train"
masked_image_dir = "C:\\Users\\gyqls\\Teamproject2\\masking_images\\data\\mask\\train"

# JSON 파일과 이미지 파일 목록 가져오기
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# color_map에 따른 색상 매핑
color_map = {
    "1": (0, 0, 255),    # Red
    "2": (0, 255, 0),    # Green
    "3": (128, 0, 128),  # Purple
    "4": (255, 0, 255),  # Magenta
    "5": (0, 128, 128),  # Teal
    "6": (128, 128, 0),  # Olive
    "7": (255, 0, 0),    # Blue
    "8": (255, 255, 0),  # Yellow
    "9": (128, 0, 0),    # Maroon
    "10": (0, 128, 0),   # Lime
    "12": (128, 128, 128), # Gray
    "13": (0, 0, 128),   # Navy
}

bbox_label_color_map = {
    "13": (255, 0, 0),  # Red
    "14": (0, 255, 0),  # Green
    "15": (0, 0, 255),  # Blue
    "16": (255, 255, 0),  # Yellow
    "17": (255, 0, 255),  # Magenta
    "18": (0, 255, 255),  # Cyan
    "19": (128, 0, 0),  # Maroon
    "20": (0, 128, 0),  # Green (Dark)
    "21": (0, 0, 128),  # Navy
    "22": (128, 128, 0),  # Olive
    "23": (128, 0, 128),  # Purple
    "24": (0, 128, 128),  # Teal
    "25": (255, 128, 0),  # Orange
    "26": (255, 0, 128),  # Pink
    "27": (128, 255, 0),  # Lime
    "28": (128, 0, 255),  # Purple (Light)
    "29": (0, 128, 255),  # Blue (Light)
    "30": (128, 128, 128),  # Gray
    "31": (192, 192, 192),  # Silver
    "32": (255, 165, 0),  # Orange (Reddish)
    "33": (255, 192, 203),  # Pink (Light)
    "35": (0, 255, 127),  # Spring Green
    "36": (218, 165, 32),  # Goldenrod
}

def find_average_coordinate(polygon_points):
    # 다각형의 x, y 좌표 추출
    x_coords = [x for x, y in polygon_points]
    y_coords = [y for x, y in polygon_points]
    # x, y 좌표의 평균 반환
    average_x = sum(x_coords) / len(x_coords)
    average_y = sum(y_coords) / len(y_coords)
    return average_x, average_y

# 이미지를 그려주는 함수 작성
def print_image(load_image, data):
    for polygon_data in data["annotations"]["environment"]:
        # polygon 데이터 좌표 추출 및 절대 좌표로 변환
        polygon_points_relative = polygon_data["points"]
        polygon_points_absolute = [(float(y), float(x)) for x, y in polygon_points_relative]
        # area_code에 따라 색상 선택
        area_code = polygon_data["area_code"]
        color = color_map.get(area_code, (0, 0, 0))  # Default to black if area_code not found in color_map

        # 다각형 그리기
        pts = np.array(polygon_points_absolute, np.int32)
        pts = pts.reshape((-1, 1, 2))
        load_image = cv2.polylines(load_image, [pts], isClosed=True, color=color, thickness=2)

        # label 데이터 추출
        area_code = polygon_data["area_code"]
        avg_x, avg_y = find_average_coordinate(polygon_points_absolute)

        # label 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"Area {area_code}"  # 라벨 텍스트
        org = (int(avg_x), int(avg_y))
        cv2.putText(load_image, label, org, font, 0.5, (0, 255, 255), 2)  # 흰색으로 표시
        
            # "PM" (바운딩 박스)에 대한 정보 처리
    for bbox_data in data["annotations"]["PM"]:
        # bbox 데이터 추출 및 절대 좌표로 변환
        bbox_info = bbox_data["points"]
        x, y, w, h = bbox_info
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))

        # 바운딩 박스 라벨 가져오기
        bbox_label = bbox_data["PM_code"]
        bbox_color = bbox_label_color_map.get(bbox_label, (0, 0, 0))  # Default to black if label not found

        # 사각형 그리기
        load_image = cv2.rectangle(load_image, top_left, bottom_right, bbox_color, thickness=2)

        # 바운딩 박스 라벨 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"Label {bbox_label}"  # 라벨 텍스트
        org = (int(x), int(y))
        cv2.putText(load_image, label, org, font, 0.5, (255, 255, 255), 2)  # 흰색으로 표시

    return load_image



# 각 JSON 파일에 대해 작업 수행
for json_filename in json_files:
    json_path = os.path.join(json_dir, json_filename)
    image_filename = os.path.splitext(json_filename)[0] + ".jpg"
    image_path = os.path.join(image_dir, image_filename)
    masked_image_filename = os.path.splitext(json_filename)[0] + "_mask.jpg"
    masked_image_path = os.path.join(masked_image_dir, masked_image_filename)

    # JSON 파일 불러오기
    with open(json_path, "r") as json_file:
        json_data = json_file.read()

    # JSON 데이터 파싱
    data = json.loads(json_data)

    # 이미지 로드
    load_image = cv2.imread(image_path)

    # 이미지에 다각형과 라벨을 그려서 얻은 결과 이미지
    image_with_labels = print_image(load_image, data)

    # 이미지 저장
    cv2.imwrite(masked_image_path, image_with_labels)