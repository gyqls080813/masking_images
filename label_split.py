import os
import random
import shutil
import uuid

def copy_with_unique_filename(src_path, dest_dir):
    base_name, ext = os.path.splitext(os.path.basename(src_path))
    unique_filename = f"{base_name}_{str(uuid.uuid4())[:8]}{ext}"
    dest_path = os.path.join(dest_dir, unique_filename)
    shutil.copy(src_path, dest_path)

# 원본 데이터 경로
data_dir = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\old_labels\\all"

# 나누어 저장할 훈련 및 검증 데이터 경로
train_dir = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\old_labels\\train"
val_dir = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\old_labels\\val"

# 데이터를 나눌 비율
train_ratio = 0.7
val_ratio = 0.3

# 데이터 파일 목록을 얻습니다.
data_files = os.listdir(data_dir)
random.shuffle(data_files)

# 훈련 및 검증 데이터 개수 계산
num_train = int(len(data_files) * train_ratio)
num_val = len(data_files) - num_train

# 훈련 데이터 복사
for file in data_files[:num_train]:
    src_path = os.path.join(data_dir, file)
    copy_with_unique_filename(src_path, train_dir)

# 검증 데이터 복사
for file in data_files[num_train:num_train+num_val]:
    src_path = os.path.join(data_dir, file)
    copy_with_unique_filename(src_path, val_dir)

print("데이터 분할이 완료되었습니다.")
