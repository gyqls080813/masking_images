from model import unet
from data import trainGenerator

data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

batch_size = 20  # 배치 크기 설정

# trainGenerator 함수를 호출하여 데이터 전처리를 진행하는 데이터 제너레이터 생성
myGenerator = trainGenerator(
    batch_size,
    'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test',  # 데이터 폴더 경로
    'images',  # 이미지 폴더 이름
    'labeling_polygon',  # 라벨 폴더 이름
    data_gen_args,
    save_to_dir='D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test\\pre_data'  # 전처리된 데이터 저장 폴더 경로
)

# 생성한 데이터 제너레이터로 이미지 전처리 및 저장 수행
num_batches = 3  # 전처리할 배치의 개수
for i, batch in enumerate(myGenerator):
    if i >= num_batches:
        break
