# 해당 파일은 polygon 데이터 중 라벨의 갯수를 측정하는 과정입니다.

import os
import json

# JSON 파일 디렉토리 경로 설정
json_dir = "C:\\Users\\gyqls\\Teamproject2\\masking_images\\labeling"

# "PM_code" 라벨 종류를 저장할 집합
pm_label_set = set()

# 디렉토리 내의 모든 JSON 파일 처리
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
        
        # "annotations" 필드에서 "PM" 라벨 추출
        if "annotations" in data and "environment" in data["annotations"]:
            for pm_data in data["annotations"]["environment"]:
                if "shape_type" in pm_data and pm_data["shape_type"] == "polygon" and "area_code" in pm_data:
                    pm_label_set.add(pm_data["area_code"])

# "PM_code" 라벨 종류를 숫자 순서대로 정렬하여 출력
sorted_pm_labels = sorted(list(pm_label_set))
print("Polygon 라벨 종류 (정렬된 순서):", sorted_pm_labels)

# Polygon 라벨 종류 (정렬된 순서): ['1', '10', '11', '12', '13', '2', '3', '4', '5', '6', '7', '8', '9']