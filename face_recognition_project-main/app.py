import os
import sqlite3
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from datetime import datetime, timedelta
import pickle
from deepface import DeepFace
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import uuid
import time



app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/documents'
app.config['ATTENDANCE_IMAGES'] = 'static/attendance_images'
app.config['FACES_FOLDER'] = 'static/faces'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ATTENDANCE_IMAGES'], exist_ok=True)
os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)


def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 email TEXT UNIQUE,
                 password TEXT,
                 name TEXT,
                 role TEXT,
                 class_name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS classes (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 class_name TEXT UNIQUE,
                 teacher_id INTEGER,
                 FOREIGN KEY (teacher_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS schedules (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 class_id INTEGER,
                 subject TEXT,
                 day_of_week TEXT,
                 start_time TEXT,
                 end_time TEXT,
                 FOREIGN KEY (class_id) REFERENCES classes(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance_windows (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 class_id INTEGER,
                 start_time TEXT,
                 end_time TEXT,
                 FOREIGN KEY (class_id) REFERENCES classes(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 class_id INTEGER,
                 date TEXT,
                 time TEXT,
                 status TEXT,
                 image_path TEXT,
                 on_time TEXT,
                 FOREIGN KEY (user_id) REFERENCES users(id),
                 FOREIGN KEY (class_id) REFERENCES classes(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 class_id INTEGER,
                 filename TEXT,
                 filepath TEXT,
                 upload_date TEXT,
                 FOREIGN KEY (class_id) REFERENCES classes(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS profiles (
                 user_id INTEGER PRIMARY KEY,
                 has_profile INTEGER,
                 FOREIGN KEY (user_id) REFERENCES users(id))''')
    c.execute("INSERT OR IGNORE INTO users (email, password, name, role) VALUES (?, ?, ?, ?)",
              ('admin@example.com', 'admin', 'Admin', 'admin'))
    conn.commit()
    conn.close()


init_db()


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password)).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['user_name'] = user['name']
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    conn = get_db_connection()
    admin_exists = conn.execute("SELECT * FROM users WHERE role = 'admin'").fetchone()
    classes = conn.execute("SELECT class_name FROM classes").fetchall()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        role = request.form['role']
        class_name = request.form.get('class_name', '')
        if role == 'student' and not class_name:
            conn.close()
            return render_template('signup.html', error='Please select a class for student role',
                                   admin_exists=admin_exists, classes=classes)
        try:
            conn.execute('INSERT INTO users (email, password, name, role, class_name) VALUES (?, ?, ?, ?, ?)',
                         (email, password, name, role, class_name))
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if role == 'student':
                conn.execute('INSERT INTO profiles (user_id, has_profile) VALUES (?, ?)', (user['id'], 0))
                conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('signup.html', error='Email already exists', admin_exists=admin_exists,
                                   classes=classes)
    conn.close()
    return render_template('signup.html', admin_exists=admin_exists, classes=classes)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    role = session['role']
    user_name = session['user_name']
    user_id = session['user_id']
    conn = get_db_connection()
    totalreg = conn.execute('SELECT COUNT(*) FROM users WHERE role = "student"').fetchone()[0]
    datetoday2 = datetime.now().strftime('%d/%m/%Y')
    if role == 'student':
        has_profile = conn.execute('SELECT has_profile FROM profiles WHERE user_id = ?', (user_id,)).fetchone()
        has_profile = has_profile['has_profile'] if has_profile else 0
    else:
        has_profile = None
    recent_attendance = conn.execute(
        'SELECT u.name, u.class_name, a.date, a.time FROM attendance a JOIN users u ON a.user_id = u.id ORDER BY a.date DESC, a.time DESC LIMIT 5').fetchall()
    names, classes, dates, times = [], [], [], []
    for record in recent_attendance:
        names.append(record['name'])
        classes.append(record['class_name'])
        dates.append(record['date'])
        times.append(record['time'])
    l = len(names)
    status_list = []
    if role == 'teacher':
        today = datetime.now().strftime('%Y-%m-%d')
        teacher_classes = conn.execute('SELECT id FROM classes WHERE teacher_id = ?', (user_id,)).fetchall()
        class_ids = [c['id'] for c in teacher_classes]
        if class_ids:
            status_list = conn.execute(
                'SELECT u.name, u.class_name, a.status, a.time, a.on_time FROM attendance a JOIN users u ON a.user_id = u.id WHERE a.class_id IN ({}) AND a.date = ?'.format(
                    ','.join('?' for _ in class_ids)), class_ids + [today]).fetchall()
            status_list = [{'name': s['name'], 'class': s['class_name'], 'status': s['status'], 'time': s['time'],
                            'on_time': s['on_time']} for s in status_list]
    conn.close()
    return render_template('home.html', role=role, user_name=user_name, totalreg=totalreg, datetoday2=datetoday2,
                           has_profile=has_profile, names=names, classes=classes, dates=dates, times=times, l=l,
                           status_list=status_list)


@app.route('/create_profile', methods=['GET', 'POST'])
def create_profile():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    profile = conn.execute('SELECT has_profile FROM profiles WHERE user_id = ?', (user_id,)).fetchone()
    if not profile or profile['has_profile']:
        conn.close()
        return redirect(url_for('home'))

    if request.method == 'POST':
        # Mở webcam để chụp ảnh khuôn mặt
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            conn.close()
            return render_template('create_profile.html', error='Cannot open webcam', nimgs=100)

        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_data = []
        i = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) < 100 and i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if k == ord('q') or len(faces_data) == 100:
                break

        video.release()
        cv2.destroyAllWindows()

        if len(faces_data) == 0:
            conn.close()
            return render_template('create_profile.html', error='No faces detected. Please try again.', nimgs=100)

        # Lưu ảnh vào thư mục static/faces/
        for idx, face in enumerate(faces_data):
            face_path = os.path.join(app.config['FACES_FOLDER'], f'{user_id}_{idx}.jpg')
            cv2.imwrite(face_path, face)
            print(f"Saved face image: {face_path}")  # Debug

        # Cập nhật trạng thái profile
        conn.execute('UPDATE profiles SET has_profile = 1 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()

        flash('Profile created successfully', 'success')
        return redirect(url_for('home'))

    conn.close()
    return render_template('create_profile.html', nimgs=100)


@app.route('/start')
def start():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()

    # Kiểm tra hồ sơ của học sinh
    profile = conn.execute('SELECT has_profile FROM profiles WHERE user_id = ?', (user_id,)).fetchone()
    if not profile or not profile['has_profile']:
        conn.close()
        return redirect(url_for('create_profile'))

    # Lấy thông tin lớp của học sinh
    user = conn.execute('SELECT class_name FROM users WHERE id = ?', (user_id,)).fetchone()
    class_name = user['class_name']
    class_info = conn.execute('SELECT id FROM classes WHERE class_name = ?', (class_name,)).fetchone()
    if not class_info:
        conn.close()
        flash('Class not found', 'error')
        return redirect(url_for('home'))

    class_id = class_info['id']

    # Kiểm tra khung giờ điểm danh
    window = conn.execute('SELECT start_time, end_time FROM attendance_windows WHERE class_id = ?',
                          (class_id,)).fetchone()
    if not window:
        conn.close()
        flash('No attendance window set for your class', 'error')
        return redirect(url_for('home'))

    # So sánh thời gian chính xác
    current_datetime = datetime.now()
    start_datetime = datetime.strptime(window['start_time'], '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(window['end_time'], '%Y-%m-%d %H:%M:%S')

    if not (start_datetime <= current_datetime <= end_datetime):
        conn.close()
        flash('Attendance window is closed', 'error')
        return redirect(url_for('home'))

    # Mở webcam để nhận diện khuôn mặt
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        conn.close()
        flash('Cannot open webcam', 'error')
        return redirect(url_for('home'))

    # Tải tất cả khuôn mặt đã đăng ký của học sinh
    known_faces = []
    face_files = [f for f in os.listdir(app.config['FACES_FOLDER']) if
                  f.startswith(f"{user_id}_") and f.endswith('.jpg')]
    for face_file in face_files:
        face_path = os.path.join(app.config['FACES_FOLDER'], face_file)
        face_image = cv2.imread(face_path)
        if face_image is not None:
            try:
                face_embedding = DeepFace.represent(face_image, model_name='VGG-Face', enforce_detection=False)[0][
                    'embedding']
                known_faces.append(face_embedding)
            except Exception as e:
                print(f"Error encoding face {face_file}: {e}")

    if not known_faces:
        video.release()
        cv2.destroyAllWindows()
        conn.close()
        flash('No face data found for this user. Please re-register your profile.', 'error')
        return redirect(url_for('create_profile'))

    # Nhận diện khuôn mặt từ webcam
    face_detected = False
    frame_count = 0
    max_frames = 300
    frame = None

    while frame_count < max_frames:
        ret, frame = video.read()
        if not ret:
            break

        try:
            # Phát hiện khuôn mặt
            face_locations = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
            for face in face_locations:
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], \
                face['facial_area']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Trích xuất embedding của khuôn mặt hiện tại
                current_face = frame[y:y + h, x:x + w]
                current_embedding = DeepFace.represent(current_face, model_name='VGG-Face', enforce_detection=False)[0][
                    'embedding']

                # So sánh với các khuôn mặt đã đăng ký
                for known_embedding in known_faces:
                    distance = np.linalg.norm(np.array(current_embedding) - np.array(known_embedding))
                    if distance < 0.6:  # Ngưỡng nhận diện
                        face_detected = True
                        break
                if face_detected:
                    break
        except Exception as e:
            print(f"Error in face detection: {e}")

        cv2.imshow('Attendance', frame)
        if face_detected:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break
        frame_count += 1

    video.release()
    cv2.destroyAllWindows()

    if not face_detected:
        conn.close()
        flash('No matching face detected. Please try again.', 'error')
        return redirect(url_for('home'))

    # Lưu ảnh điểm danh
    image_filename = f"{user_id}_{uuid.uuid4()}.jpg"
    image_path = os.path.join(app.config['ATTENDANCE_IMAGES'], image_filename)
    cv2.imwrite(image_path, frame)

    # Ghi lại điểm danh vào cơ sở dữ liệu
    today = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H:%M')
    on_time = 'Yes' if current_datetime <= start_datetime + timedelta(minutes=5) else 'No'
    status = 'Present'

    conn.execute(
        'INSERT INTO attendance (user_id, class_id, date, time, status, image_path, on_time) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, class_id, today, current_time_str, status, image_path, on_time))
    conn.commit()
    conn.close()

    flash('Attendance recorded successfully', 'success')
    return redirect(url_for('home'))


@app.route('/status')
def status():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    today = datetime.now().strftime('%Y-%m-%d')
    teacher_classes = conn.execute('SELECT id FROM classes WHERE teacher_id = ?', (user_id,)).fetchall()
    class_ids = [c['id'] for c in teacher_classes]
    if not class_ids:
        conn.close()
        return render_template('status_table.html', status_list=[])
    status_list = conn.execute(
        'SELECT u.name, u.class_name, a.status, a.time, a.on_time, a.image_path FROM attendance a JOIN users u ON a.user_id = u.id WHERE a.class_id IN ({}) AND a.date = ?'.format(
            ','.join('?' for _ in class_ids)), class_ids + [today]).fetchall()
    status_list = [
        {'name': s['name'], 'class': s['class_name'], 'status': s['status'], 'time': s['time'], 'on_time': s['on_time'],
         'image_path': s['image_path']} for s in status_list]
    conn.close()
    return render_template('status_table.html', status_list=status_list)


@app.route('/manage_schedule', methods=['GET', 'POST'])
def manage_schedule():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    classes = conn.execute('SELECT id, class_name FROM classes WHERE teacher_id = ?', (user_id,)).fetchall()
    if request.method == 'POST':
        class_id = request.form['class_id']
        subject = request.form['subject']
        day_of_week = request.form['day_of_week']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        if start_time >= end_time:
            conn.close()
            return render_template('manage_schedule.html', error='End time must be after start time', classes=classes,
                                   schedules=[])
        conn.execute(
            'INSERT INTO schedules (class_id, subject, day_of_week, start_time, end_time) VALUES (?, ?, ?, ?, ?)',
            (class_id, subject, day_of_week, start_time, end_time))
        conn.commit()
    schedules = conn.execute(
        'SELECT c.class_name, s.subject, s.day_of_week, s.start_time, s.end_time FROM schedules s JOIN classes c ON s.class_id = c.id WHERE c.teacher_id = ?',
        (user_id,)).fetchall()
    schedules = [{'class_name': s['class_name'], 'subject': s['subject'], 'day_of_week': s['day_of_week'],
                  'start_time': s['start_time'], 'end_time': s['end_time']} for s in schedules]
    conn.close()
    return render_template('manage_schedule.html', classes=classes, schedules=schedules)


@app.route('/manage_attendance_window', methods=['GET', 'POST'])
def manage_attendance_window():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    classes = conn.execute('SELECT id, class_name FROM classes WHERE teacher_id = ?', (user_id,)).fetchall()

    if request.method == 'POST':
        class_id = request.form['class_id']
        start_time = request.form['start_time']  # Định dạng từ input: 'YYYY-MM-DDTHH:MM'
        end_time = request.form['end_time']

        try:
            # Chuyển đổi định dạng từ 'YYYY-MM-DDTHH:MM' sang 'YYYY-MM-DD HH:MM:SS'
            start_datetime = datetime.strptime(start_time, '%Y-%m-%dT%H:%M')
            end_datetime = datetime.strptime(end_time, '%Y-%m-%dT%H:%M')
            start_time_formatted = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            end_time_formatted = end_datetime.strftime('%Y-%m-%d %H:%M:%S')

            if start_datetime >= end_datetime:
                conn.close()
                return render_template('manage_attendance_window.html', error='End time must be after start time',
                                       classes=classes, windows=[])
        except ValueError:
            conn.close()
            return render_template('manage_attendance_window.html',
                                   error='Invalid time format. Use YYYY-MM-DD HH:MM:SS', classes=classes, windows=[])

        conn.execute('INSERT OR REPLACE INTO attendance_windows (class_id, start_time, end_time) VALUES (?, ?, ?)',
                     (class_id, start_time_formatted, end_time_formatted))
        conn.commit()

    windows = conn.execute(
        'SELECT c.class_name, w.start_time, w.end_time FROM attendance_windows w JOIN classes c ON w.class_id = c.id WHERE c.teacher_id = ?',
        (user_id,)).fetchall()
    windows = [{'class_name': w['class_name'], 'start_time': w['start_time'], 'end_time': w['end_time']} for w in
               windows]
    conn.close()
    return render_template('manage_attendance_window.html', classes=classes, windows=windows)


@app.route('/manage_documents', methods=['GET', 'POST'])
def manage_documents():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    classes = conn.execute('SELECT id, class_name FROM classes WHERE teacher_id = ?', (user_id,)).fetchall()
    if request.method == 'POST':
        class_id = request.form['class_id']
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            upload_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute('INSERT INTO documents (class_id, filename, filepath, upload_date) VALUES (?, ?, ?, ?)',
                         (class_id, filename, filepath, upload_date))
            conn.commit()
    documents = conn.execute(
        'SELECT c.class_name, d.filename, d.filepath, d.upload_date FROM documents d JOIN classes c ON d.class_id = c.id WHERE c.teacher_id = ?',
        (user_id,)).fetchall()
    documents = [{'class_name': d['class_name'], 'filename': d['filename'], 'filepath': d['filepath'],
                  'upload_date': d['upload_date']} for d in documents]
    conn.close()
    return render_template('manage_documents.html', classes=classes, documents=documents)


@app.route('/download_document/<path:filepath>')
def download_document(filepath):
    return send_file(filepath, as_attachment=True)


@app.route('/view_schedule')
def view_schedule():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    user = conn.execute('SELECT class_name FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return render_template('view_schedule.html', error='User not found')
    class_name = user['class_name']
    class_info = conn.execute('SELECT id FROM classes WHERE class_name = ?', (class_name,)).fetchone()
    if not class_info:
        conn.close()
        return render_template('view_schedule.html', error='Class not found')
    class_id = class_info['id']
    schedules = conn.execute('SELECT subject, day_of_week, start_time, end_time FROM schedules WHERE class_id = ?',
                             (class_id,)).fetchall()
    schedules = [{'subject': s['subject'], 'day_of_week': s['day_of_week'], 'start_time': s['start_time'],
                  'end_time': s['end_time']} for s in schedules]
    conn.close()
    return render_template('view_schedule.html', schedules=schedules)


@app.route('/view_documents')
def view_documents():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    user = conn.execute('SELECT class_name FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return render_template('view_documents.html', error='User not found')
    class_name = user['class_name']
    class_info = conn.execute('SELECT id FROM classes WHERE class_name = ?', (class_name,)).fetchone()
    if not class_info:
        conn.close()
        return render_template('view_documents.html', error='Class not found')
    class_id = class_info['id']
    documents = conn.execute('SELECT filename, filepath, upload_date FROM documents WHERE class_id = ?',
                             (class_id,)).fetchall()
    documents = [{'filename': d['filename'], 'filepath': d['filepath'], 'upload_date': d['upload_date']} for d in
                 documents]
    conn.close()
    return render_template('view_documents.html', documents=documents)


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    conn = get_db_connection()
    classes = conn.execute(
        'SELECT c.class_name, u.name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.id').fetchall()
    classes = [{'class_name': c['class_name'], 'teacher_name': c['teacher_name']} for c in classes]
    teachers = conn.execute('SELECT id, name FROM users WHERE role = "teacher"').fetchall()
    conn.close()
    return render_template('admin_dashboard.html', classes=classes, teachers=teachers)


@app.route('/create_class', methods=['GET', 'POST'])
def create_class():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    conn = get_db_connection()
    teachers = conn.execute('SELECT id, name FROM users WHERE role = "teacher"').fetchall()
    if request.method == 'POST':
        class_name = request.form['class_name']
        teacher_id = request.form['teacher_id']
        try:
            conn.execute('INSERT INTO classes (class_name, teacher_id) VALUES (?, ?)', (class_name, teacher_id))
            conn.commit()
            conn.close()
            return redirect(url_for('admin_dashboard'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('create_class.html', error='Class name already exists', teachers=teachers)
    conn.close()
    return render_template('create_class.html', teachers=teachers)


@app.route('/report', methods=['GET', 'POST'])
def report():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    classes = conn.execute('SELECT id, class_name FROM classes WHERE teacher_id = ?', (user_id,)).fetchall()
    selected_class = 'all'
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        selected_class = request.form.get('class_name', 'all')
        class_ids = [c['id'] for c in classes]
        query = 'SELECT u.name, u.class_name, a.date, a.status, a.on_time FROM attendance a JOIN users u ON a.user_id = u.id WHERE a.class_id IN ({})'.format(
            ','.join('?' for _ in class_ids))
        params = class_ids
        if selected_class != 'all':
            class_info = conn.execute('SELECT id FROM classes WHERE class_name = ?', (selected_class,)).fetchone()
            if class_info:
                query += ' AND a.class_id = ?'
                params.append(class_info['id'])
        if start_date:
            query += ' AND a.date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND a.date <= ?'
            params.append(end_date)
        data = conn.execute(query, params).fetchall()
        if not data:
            conn.close()
            return render_template('report.html', error='No data found for the selected criteria', classes=classes,
                                   selected_class=selected_class)
        df = pd.DataFrame(data, columns=['name', 'class_name', 'date', 'status', 'on_time'])
        plt.figure(figsize=(10, 6))
        sns.countplot(x='date', hue='status', data=df)
        plt.xticks(rotation=45)
        plt.title('Attendance Status Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot = base64.b64encode(img.getvalue()).decode()
        plt.close()
        session['report_data'] = df.to_dict()
        conn.close()
        return render_template('report.html', plot=plot, classes=classes, selected_class=selected_class)
    conn.close()
    return render_template('report.html', classes=classes, selected_class=selected_class)


@app.route('/download_report')
def download_report():
    if 'report_data' not in session:
        return redirect(url_for('report'))
    df = pd.DataFrame(session['report_data'])
    csv = df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='attendance_report.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)