# 해당 폴더는 들어오는 이미지의 크기를 확인하는 과정입니다.

import cv2

# 경로를 실제 이미지 파일 경로로 수정해주세요
image_path = "C:\\Users\\gyqls\\Teamproject2\\masking_images\\data\\images\\val\\C000067_007_0220_C_D_F_0.jpg"

# 이미지를 읽어서 크기를 확인
image = cv2.imread(image_path)
if image is not None:
    height, width, channels = image.shape
    print("Image width:", width)
    print("Image height:", height)
else:
    print("Failed to load the image.")
