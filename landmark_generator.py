# get landmarks from an array of image paths
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #make sure this file is in the correct path
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

def get_landmarks(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    all_landmarks = []

    if len(faces) == 0:
        gray = cv2.equalizeHist(gray)
        faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            all_landmarks.append([x, y])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    all_landmarks = np.array(all_landmarks)
    norm_landmarks = (all_landmarks - all_landmarks.min()) / (all_landmarks.max() - all_landmarks.min())
    mouth_landmarks = norm_landmarks[48:68]
    return mouth_landmarks


def plot_landmarks(landmarks):
    x_coords = [landmark[0] for landmark in landmarks]
    y_coords = [landmark[1] for landmark in landmarks]

    plt.scatter(x_coords, y_coords)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Facial Landmarks")
    plt.savefig('LandmarkPlot.png')


def main():
    if len(sys.argv) != 2:
        print("Usage: python file.py <input>")
        sys.exit(1)

    path_arr = sys.argv[1]
    landmark_output = []
    for path in path_arr:
        landmark_output.append(get_landmarks(path))

    with open('landmark_output.txt', 'w') as file:
        file.write(landmark_output)

    print(f"Saved array of ${len(landmark_output)} mouth landmarks")


if __name__ == '__main__':
    main()


