import cv2

# 경로를 실제 이미지 파일 경로로 수정해주세요
image_path = "D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\data\\test\\image\\C000067_007_0220_C_D_F_0.jpg"

# 이미지를 읽어서 크기를 확인
image = cv2.imread(image_path)
if image is not None:
    height, width, channels = image.shape
    print("Image width:", width)
    print("Image height:", height)
else:
    print("Failed to load the image.")
