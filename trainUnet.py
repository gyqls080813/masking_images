from model import unet
from data import trainGenerator
from keras.callbacks import ModelCheckpoint  # ModelCheckpoint를 import 추가

data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# trainGenerator 함수를 호출하여 데이터 전처리를 진행하는 데이터 제너레이터 생성
myGenerator = trainGenerator(
    2,
    'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test',  # 수정: 학습 데이터 폴더 경로
    'images',  # 이미지 폴더 이름
    'labeling_polygon',  # 라벨 폴더 이름
    data_gen_args,
    save_to_dir='D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test\\pre_data'  # 수정: 전처리된 데이터 저장 폴더 경로
)
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit(myGenerator, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])  # 수정: myGene -> myGenerator
