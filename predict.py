from model import unet
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# 예측 및 결과 저장 함수
def predict_and_save(model, image_path, output_dir, threshold=0.5):
    # 이미지 불러오기 및 전처리
    img = load_img(image_path, color_mode='grayscale', target_size=(256, 256))
    img_array = img_to_array(img) / 255.0  # 이미지를 0~1 사이 값으로 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    # 예측 수행
    prediction = model.predict(img_array)

    # 이진화된 예측 결과 생성
    binary_prediction = (prediction > threshold).astype(np.uint8)

    # 결과 이미지를 저장할 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 결과 이미지 파일명
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))

    # 결과 이미지 저장
    binary_prediction_img = Image.fromarray(np.squeeze(binary_prediction) * 255)
    binary_prediction_img.save(output_image_path)

    # 시각화 (원본 이미지와 예측 결과 마스크)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(img_array), cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(binary_prediction), cmap='gray')
    plt.title('Predicted Mask')

    plt.savefig(os.path.join(output_dir, 'visualization.png'))
    plt.close()

# 모델 불러오기
model = unet()
model.load_weights('unet_membrane.hdf5')  # 미리 학습된 모델의 가중치 불러오기

# 예측할 이미지가 있는 디렉토리 경로
input_directory = r'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test\\val_image'

# 결과 이미지를 저장할 디렉토리 경로
output_main_directory = r'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test\\val_result_image'

# 예측 및 결과 저장
for image_filename in os.listdir(input_directory):
    if image_filename.endswith('.jpg'):
        image_path = os.path.join(input_directory, image_filename)
        output_subdir = os.path.join(output_main_directory, 'result_' + image_filename[:-4])
        predict_and_save(model, image_path, output_subdir)
