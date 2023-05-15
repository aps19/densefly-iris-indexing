from facenet_pytorch import MTCNN
import torch
import cv2
import numpy as np
import glob
import os
import json

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

def get_images(path):
    imgs = []
    label = 0

    for filepath in glob.iglob(path):
        num_in_folder = 0
        for file in glob.iglob(filepath+'/L/*'):
            if file.endswith('.jpg'):
                img = preprocess_image(file)
                imgs.append([img, num_in_folder, label, img])
                num_in_folder += 1

        for file in glob.iglob(filepath+'/R/*'):
            if file.endswith('.jpg'):
                img = preprocess_image(file)
                imgs.append([img, num_in_folder, label, img])
                num_in_folder += 1

        label += 1
    return imgs


def preprocess_image(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (200, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def detect_eyes(imgs):
    eye_detected = []
    for i, num, label, c in imgs:
        i = cv2.resize(i, (400, 300))
        boxes, probs, landmarks = mtcnn.detect(i, landmarks=True)
        # Other code here
        eye_detected.append([i, num, label, c])

        # Save eye image
        filename = f'eye_images/{label}_{num}.jpg'
        cv2.imwrite(filename, cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

        # Add metadata to list
        metadata.append({
            'filename': filename,
            'label': label,
            'shape': i.shape,
        })

    return eye_detected


def detect_iris(eye_detected):
    iris_eye_detected = []
    for i, num, label, c in eye_detected:
        circles = cv2.HoughCircles(i, cv2.HOUGH_GRADIENT, 10, 100)
        # Other code here
        iris_eye_detected.append([i, num, label, c])

        # Save iris image
        filename = f'iris_images/{label}_{num}.jpg'
        cv2.imwrite(filename, cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

        # Add metadata to list
        metadata.append({
            'filename': filename,
            'label': label,
            'shape': i.shape,
        })

    return iris_eye_detected


def main():
    # Define your path here
    path = 'CASIA-Iris-Thousand/*'
    imgs = get_images(path)
    print("total images number ", len(imgs))

    global metadata
    metadata = []

    eye_detected = detect_eyes(imgs)
    print("total eyes found = ", len(eye_detected))

    iris_eye_detected = detect_iris(eye_detected)
    print("total iris found = ", len(iris_eye_detected))

    # Save metadata to JSON file
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)

    # Other code here


if __name__ == "__main__":
    main()
