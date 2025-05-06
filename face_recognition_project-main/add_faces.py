import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

# Đảm bảo thư mục data tồn tại
if not os.path.exists('data'):
    os.makedirs('data')

names_file = 'data/names.pkl'
faces_file = 'data/faces_data.pkl'

# Kiểm tra đồng bộ giữa names.pkl và faces_data.pkl
if os.path.isfile(names_file) and os.path.isfile(faces_file):
    with open(names_file, 'rb') as f:
        existing_names = pickle.load(f)
    with open(faces_file, 'rb') as f:
        existing_faces = pickle.load(f)

    # Kiểm tra số lượng nhãn và mẫu
    if len(existing_names) != existing_faces.shape[0]:
        print("Warning: Inconsistent data detected. Resetting data files...")
        existing_names = []
        existing_faces = np.empty((0, faces_data.shape[1]), dtype=faces_data.dtype)
else:
    existing_names = []
    existing_faces = np.empty((0, faces_data.shape[1]), dtype=faces_data.dtype)

# Cập nhật dữ liệu
new_names = [name] * len(faces_data)
existing_names.extend(new_names)
existing_faces = np.append(existing_faces, faces_data, axis=0)

# Lưu dữ liệu
with open(names_file, 'wb') as f:
    pickle.dump(existing_names, f)
with open(faces_file, 'wb') as f:
    pickle.dump(existing_faces, f)

# Kiểm tra lại
with open(names_file, 'rb') as f:
    names = pickle.load(f)
with open(faces_file, 'rb') as f:
    faces = pickle.load(f)

print(f"Number of labels: {len(names)}")
print(f"Number of samples: {faces.shape[0]}")
if len(names) != faces.shape[0]:
    raise ValueError("Mismatch between number of labels and samples after saving!")