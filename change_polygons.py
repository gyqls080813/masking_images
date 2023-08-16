import os
import json
import shutil

# 입력과 출력 디렉토리 경로 설정
input_directory = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\old_labels"
output_directory = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\polygon_labels"

# 입력 디렉토리의 "val"과 "train" 디렉토리에 대해 처리
for subset in ["all"]:
    input_subset_dir = os.path.join(input_directory, subset)
    output_subset_dir = os.path.join(output_directory, subset)
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_subset_dir, exist_ok=True)
    
    # "val" 또는 "train" 디렉토리 내의 JSON 파일에 대해 처리
    for filename in os.listdir(input_subset_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_subset_dir, filename)
            output_path = os.path.join(output_subset_dir, filename)
            
            # JSON 파일 열기
            with open(input_path, "r") as f:
                data = json.load(f)
            
            # "annotations" 키 아래의 "environment" 리스트에서 "shape_type" 값이 "polygon"인 항목 추출
            environment_annotations = [item for item in data.get("annotations", {}).get("environment", []) if item.get("shape_type") == "polygon"]
            
            # 추출된 결과를 새로운 파일로 저장
            with open(output_path, "w") as f:
                json.dump({"environment": environment_annotations}, f, indent=2)
            
            print(f"Processed {input_path}, saved environment polygons to {output_path}")

print("Done")
