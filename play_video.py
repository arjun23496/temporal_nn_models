import cv2
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture("/Users/arjunkaruvally/Downloads/filezilla/video.avi")

success, image_sample = vidcap.read()

while success:
    success, image_sample = vidcap.read()
    plt.imshow(image_sample)
    plt.show()
