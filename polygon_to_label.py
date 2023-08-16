import os
import json

# 원본 JSON 파일이 있는 디렉토리 경로
source_dir = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\old_labels\\all"

# 수정된 JSON 파일을 저장할 디렉토리 경로
destination_dir = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\labeling_poly\\polygon_bbox"

# 새로운 라벨 매핑
new_labels = {
    "1": "sidewalk",
    "2": "crosswalk",
    "3": "bikeroads",
    "4": "crossroads",
    "5": "centerline",
    "6": "num1",
    "7": "stopline",
    "8": "Between",
    "9": "green_crosswalk",
    "10": "red_crosswalk",
    "11": "green_driveway",
    "12": "red_driveway",
    "13": "num2",
}

# source_dir 내의 모든 파일 목록을 가져오기
json_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

# 각 JSON 파일을 읽어서 수정 후 저장
for json_file in json_files:
    json_path = os.path.join(source_dir, json_file)

    # JSON 파일 읽기
    with open(json_path, "r") as f:
        data = json.load(f)

    # environment 섹션의 annotations 수정
    for annotation in data["annotations"]["environment"]:
        area_code = annotation["area_code"]
        new_label = new_labels.get(area_code)
        if new_label:
            annotation["area_code"] = new_label

    # 수정된 JSON 데이터를 새로운 디렉토리에 저장
    new_json_path = os.path.join(destination_dir, json_file)
    with open(new_json_path, "w") as f:
        json.dump(data, f, indent=4)

print("데이터 변환 및 저장이 완료되었습니다.")