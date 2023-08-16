import os
import json

# JSON 파일 디렉토리 경로 설정
json_dir = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\old_labels\\all"

# "PM_code" 라벨 종류를 저장할 집합
pm_label_set = set()

# 디렉토리 내의 모든 JSON 파일 처리
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
        
        # "annotations" 필드에서 "PM" 라벨 추출
        if "annotations" in data and "PM" in data["annotations"]:
            for pm_data in data["annotations"]["PM"]:
                if "shape_type" in pm_data and pm_data["shape_type"] == "bbox" and "PM_code" in pm_data:
                    pm_label_set.add(pm_data["PM_code"])

# "PM_code" 라벨 종류를 숫자 순서대로 정렬하여 출력
sorted_pm_labels = sorted(list(pm_label_set))
print("PM 라벨 종류 (정렬된 순서):", sorted_pm_labels)

# PM 라벨 종류 (정렬된 순서): ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '35', '36']