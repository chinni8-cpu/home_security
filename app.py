from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pymysql
import threading
import time
import numpy as np
import re
import urllib.parse

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/cctv_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads/family_members'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# WhatsApp Configuration
WHATSAPP_NUMBER = '919876543210'  # Replace with your WhatsApp number (with country code)
WHATSAPP_MESSAGE = '🚨 ALERT: Unknown person detected in CCTV camera! Please check immediately.'

# Email Configuration (for mailto link)
EMAIL_TO = 'your-email@gmail.com'  # Replace with your email
EMAIL_SUBJECT = '🚨 CCTV Alert: Unknown Person Detected'
EMAIL_BODY = 'Dear Team,\n\nAn unknown person has been detected in the CCTV camera. Please check the system immediately.\n\nTimestamp: {timestamp}\n\nRegards,\nCCTV Monitoring System'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FamilyMember(db.Model):
    __tablename__ = 'family_members'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Alert(db.Model):
    __tablename__ = 'alerts'
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50))
    message = db.Column(db.Text)
    image_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()


# ==============================================
# LBPH FACE RECOGNIZER (WITH DATA AUGMENTATION)
# Generates multiple synthetic samples from a single photo
# to drastically improve accuracy and prevent false matches.
# ==============================================
class SimpleFaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.lbph = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
        self.lbph_trained = False
        self.name_map = {}
        self.known_face_names = []
        self.face_locations = []
        self.face_names = []
        self.face_scores = []
        
        # 145.0 represents exactly a 50% similarity match requirement
        self.CONFIDENCE_THRESHOLD = 145.0
        self.load_known_faces()

    def detect_face(self, gray_img):
        """Detect the largest face in a grayscale image. Returns face ROI or None."""
        for scale in [1.05, 1.1, 1.2, 1.3]:
            faces = self.face_cascade.detectMultiScale(
                gray_img, scaleFactor=scale, minNeighbors=4, minSize=(50, 50)
            )
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                pad = int(0.08 * w)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(gray_img.shape[1], x + w + pad)
                y2 = min(gray_img.shape[0], y + h + pad)
                face_roi = gray_img[y1:y2, x1:x2]
                return cv2.resize(face_roi, (100, 100))
        return None

    def generate_synthetic_samples(self, face_roi):
        """Create multiple variations of a face to strengthen the LBPH model."""
        samples = []
        base = cv2.resize(face_roi, (100, 100))
        samples.append(base)
        # Flip horizontal
        samples.append(cv2.flip(base, 1))
        # Brightness variations
        samples.append(cv2.convertScaleAbs(base, alpha=1.2, beta=15))
        samples.append(cv2.convertScaleAbs(base, alpha=0.8, beta=-15))
        # Small crops to simulate distance changes
        h, w = base.shape
        samples.append(cv2.resize(base[5:h, 5:w], (100, 100)))
        samples.append(cv2.resize(base[0:h-5, 0:w-5], (100, 100)))
        samples.append(cv2.resize(base[5:h-5, 5:w-5], (100, 100)))
        return samples

    def _retrain(self):
        if not self.face_samples:
            self.lbph_trained = False
            return
        faces_arr = [np.array(f, dtype=np.uint8) for f in self.face_samples]
        labels_arr = np.array(self.face_labels, dtype=np.int32)
        self.lbph.train(faces_arr, labels_arr)
        self.lbph_trained = True
        print(f"[INFO] LBPH trained on {len(faces_arr)} synthetic samples across {len(self.name_map)} people")

    def load_known_faces(self):
        with app.app_context():
            family_members = FamilyMember.query.all()

        print(f"[INFO] Loading {len(family_members)} family members into LBPH...")
        self.name_map = {}
        self.known_face_names = []
        self.face_samples = []
        self.face_labels = []

        for member in family_members:
            if not (member.image_path and os.path.exists(member.image_path)):
                continue

            img = cv2.imread(member.image_path)
            if img is None: continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_roi = self.detect_face(gray)

            if face_roi is None:
                h_img, w_img = gray.shape[:2]
                crop = gray[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
                if crop.size > 0:
                    face_roi = cv2.resize(crop, (100, 100))
                else:
                    continue

            # Data Augmentation: Generate 7 samples from 1 photo
            samples = self.generate_synthetic_samples(face_roi)
            for s in samples:
                self.face_samples.append(s)
                self.face_labels.append(member.id)
                
            self.name_map[member.id] = member.name
            if member.name not in self.known_face_names:
                self.known_face_names.append(member.name)
            print(f"  [OK] Loaded & Augmented: {member.name}")

        self._retrain()

    def reload_known_faces(self):
        self.load_known_faces()

    def add_family_member(self, name, image_path, member_id):
        if not os.path.exists(image_path):
            return False
        img = cv2.imread(image_path)
        if img is None: return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = self.detect_face(gray)

        if face_roi is None:
            h_img, w_img = gray.shape[:2]
            crop = gray[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
            if crop.size > 0:
                face_roi = cv2.resize(crop, (100, 100))
            else:
                return False

        samples = self.generate_synthetic_samples(face_roi)
        for s in samples:
            self.face_samples.append(s)
            self.face_labels.append(member_id)

        self.name_map[member_id] = name
        if name not in self.known_face_names:
            self.known_face_names.append(name)

        self._retrain()
        return True

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        self.face_locations = []
        self.face_names = []
        self.face_scores = []
        unknown_detected = False

        for (x, y, w, h) in faces:
            self.face_locations.append((y, x + w, y + h, x))
            face_roi = gray[y:y + h, x:x + w]

            if face_roi.size == 0 or not self.lbph_trained:
                self.face_names.append("Unknown")
                self.face_scores.append(999.0)
                unknown_detected = True
                continue

            face_resized = cv2.resize(face_roi, (100, 100))

            try:
                label, confidence = self.lbph.predict(face_resized)
            except Exception:
                self.face_names.append("Unknown")
                self.face_scores.append(999.0)
                unknown_detected = True
                continue

            if confidence <= self.CONFIDENCE_THRESHOLD:
                name = self.name_map.get(label, "Unknown")
                self.face_names.append(name)
                self.face_scores.append(confidence)
                print(f"  [MATCH] {name}  (conf={confidence:.1f})")
            else:
                self.face_names.append("Unknown")
                self.face_scores.append(confidence)
                unknown_detected = True
                print(f"  [UNKNOWN] conf={confidence:.1f} > threshold={self.CONFIDENCE_THRESHOLD}")

        return self.face_names, unknown_detected

    def get_face_positions(self):
        return self.face_locations

    def get_face_scores(self):
        return self.face_scores



# Initialize face recognizer
face_recognizer = SimpleFaceRecognizer()

# Camera control
camera_active = False
camera_thread = None
frame_buffer = None
alert_messages = []

# Live status counters (updated by camera_loop)
live_status = {
    "total_faces": 0,
    "known_faces": 0,
    "unknown_faces": 0,
    "fps": 0.0,
    "camera_on": False
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def draw_face_overlay(frame, face_locations, face_names, face_scores):
    """
    Draw premium face detection overlays:
    - Green rounded box + name badge for known family members
    - Red pulsing box + ⚠ UNKNOWN warning for strangers
    """
    h_frame, w_frame = frame.shape[:2]

    for idx, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
        score = face_scores[idx] if idx < len(face_scores) else 0.0
        is_unknown = (name == "Unknown")

        # --- Colors ---
        if is_unknown:
            box_color   = (0, 0, 220)      # Red
            badge_color = (0, 0, 180)
            text_color  = (255, 255, 255)
            glow_color  = (0, 0, 255)
        else:
            box_color   = (0, 210, 60)     # Green
            badge_color = (0, 160, 40)
            text_color  = (255, 255, 255)
            glow_color  = (0, 255, 100)

        thickness = 3

        # --- Outer glow (semi-transparent) ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (left - 4, top - 4), (right + 4, bottom + 4), glow_color, 4)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # --- Main bounding box ---
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)

        # --- Corner accents (futuristic) ---
        corner_len = min(20, (right - left) // 4)
        # Top-left
        cv2.line(frame, (left, top), (left + corner_len, top), box_color, thickness + 1)
        cv2.line(frame, (left, top), (left, top + corner_len), box_color, thickness + 1)
        # Top-right
        cv2.line(frame, (right, top), (right - corner_len, top), box_color, thickness + 1)
        cv2.line(frame, (right, top), (right, top + corner_len), box_color, thickness + 1)
        # Bottom-left
        cv2.line(frame, (left, bottom), (left + corner_len, bottom), box_color, thickness + 1)
        cv2.line(frame, (left, bottom), (left, bottom - corner_len), box_color, thickness + 1)
        # Bottom-right
        cv2.line(frame, (right, bottom), (right - corner_len, bottom), box_color, thickness + 1)
        cv2.line(frame, (right, bottom), (right, bottom - corner_len), box_color, thickness + 1)

        # --- Name badge background ---
        label = f"⚠ UNKNOWN" if is_unknown else name
        font      = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        font_thick = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        badge_x1 = left
        badge_y1 = max(0, top - text_h - 16)
        badge_x2 = left + text_w + 16
        badge_y2 = top

        badge_overlay = frame.copy()
        cv2.rectangle(badge_overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_color, -1)
        cv2.addWeighted(badge_overlay, 0.85, frame, 0.15, 0, frame)

        # --- Name / UNKNOWN text ---
        cv2.putText(
            frame, label,
            (badge_x1 + 8, badge_y2 - 6),
            font, font_scale, text_color, font_thick, cv2.LINE_AA
        )

        # --- Confidence bar (for known members) ---
        if not is_unknown and score > 0:
            conf_pct = min(100, int(score * 100))
            bar_x = left
            bar_y = bottom + 6
            bar_w = right - left
            bar_h = 5
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            # Fill
            fill = int((conf_pct / 100) * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), box_color, -1)
            cv2.putText(
                frame, f"{conf_pct}%",
                (bar_x + bar_w + 5, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA
            )

        # --- UNKNOWN extra warning text on face ---
        if is_unknown:
            warn_font  = cv2.FONT_HERSHEY_SIMPLEX
            warn_scale = 0.5
            warn_text  = "INTRUDER ALERT"
            (ww, wh), _ = cv2.getTextSize(warn_text, warn_font, warn_scale, 1)
            wx = left + (right - left - ww) // 2
            wy = top + (bottom - top) // 2 + wh // 2
            # Dark backing
            cv2.rectangle(frame, (wx - 4, wy - wh - 4), (wx + ww + 4, wy + 4), (0, 0, 0), -1)
            cv2.putText(frame, warn_text, (wx, wy), warn_font, warn_scale, (0, 80, 255), 1, cv2.LINE_AA)


def draw_hud(frame, total_faces, known_count, unknown_count, fps):
    """Draw top & bottom HUD banners with live stats"""
    h, w = frame.shape[:2]

    # --- Top banner ---
    top_overlay = frame.copy()
    cv2.rectangle(top_overlay, (0, 0), (w, 60), (10, 10, 25), -1)
    cv2.addWeighted(top_overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, 60), (w, 60), (0, 180, 255), 1)

    cv2.putText(frame, "CCTV FAMILY SECURITY SYSTEM",
                (14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 120, 38), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)

    # --- Bottom stats banner ---
    bot_overlay = frame.copy()
    cv2.rectangle(bot_overlay, (0, h - 50), (w, h), (10, 10, 25), -1)
    cv2.addWeighted(bot_overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, h - 50), (w, h - 50), (0, 180, 255), 1)

    # Stats
    cv2.putText(frame, f"FACES: {total_faces}",
                (14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"KNOWN: {known_count}",
                (130, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 60), 1, cv2.LINE_AA)

    unknown_color = (0, 80, 255) if unknown_count > 0 else (200, 200, 200)
    cv2.putText(frame, f"UNKNOWN: {unknown_count}",
                (260, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, unknown_color, 1, cv2.LINE_AA)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts,
                (w - 90, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


# ==============================================
# SERVE UPLOADED FILES
# ==============================================
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve files from uploads folder"""
    try:
        safe_path = os.path.join('uploads', filename)
        if not os.path.exists(safe_path):
            return jsonify({'error': f'File {filename} not found'}), 404
        return send_from_directory('uploads', filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


# ==============================================
# DEBUG ENDPOINTS
# ==============================================
@app.route('/debug')
def debug_files():
    upload_path = app.config['UPLOAD_FOLDER']
    files = []
    if os.path.exists(upload_path):
        files = os.listdir(upload_path)
    return jsonify({
        'folder': upload_path,
        'files': files,
        'file_count': len(files),
        'folder_exists': os.path.exists(upload_path),
        'absolute_path': os.path.abspath(upload_path)
    })

@app.route('/debug_face_recognition')
def debug_face_recognition():
    result = {
        'known_faces': len(face_recognizer.known_face_names),
        'known_names': face_recognizer.known_face_names,
        'lbph_trained': face_recognizer.lbph_trained,
        'confidence_threshold': face_recognizer.CONFIDENCE_THRESHOLD,
        'message': ''
    }
    with app.app_context():
        members = FamilyMember.query.all()
        result['db_members'] = [{'name': m.name, 'path': m.image_path} for m in members]
        for member in members:
            if member.image_path and os.path.exists(member.image_path):
                img = cv2.imread(member.image_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_recognizer.face_cascade.detectMultiScale(gray, 1.1, 3)
                    result[f'{member.name}_faces_detected'] = len(faces)
                else:
                    result[f'{member.name}_error'] = 'Could not load image'
            else:
                result[f'{member.name}_error'] = 'File not found'
    return jsonify(result)


# ==============================================
# FACE STATUS (live stats for dashboard)
# ==============================================
@app.route('/face_status')
def face_status():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(live_status)


# ==============================================
# AUTH ROUTES
# ==============================================
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password, email=email)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])


# ==============================================
# FAMILY MEMBER ROUTES
# ==============================================
@app.route('/add_family_member', methods=['POST'])
def add_family_member():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    name = request.form['name']

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        name_clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], name_clean)

        counter = 1
        while os.path.exists(filepath):
            name_parts = name_clean.rsplit('.', 1)
            if len(name_parts) == 2:
                name_clean = f"{name_parts[0]}_{counter}.{name_parts[1]}"
            else:
                name_clean = f"{name_clean}_{counter}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], name_clean)
            counter += 1

        file.save(filepath)

        new_member = FamilyMember(name=name, image_path=filepath)
        db.session.add(new_member)
        db.session.commit()

        # Pass member_id so LBPH uses the DB id as the integer label
        success = face_recognizer.add_family_member(name, filepath, new_member.id)
        msg = 'Family member added and recognised!' if success else 'Member added (face not detected — try a clearer front-facing photo)'

        return jsonify({'success': True, 'message': msg})

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/delete_family_member/<int:member_id>', methods=['DELETE'])
def delete_family_member(member_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        member = FamilyMember.query.get(member_id)
        if not member:
            return jsonify({'error': 'Member not found'}), 404

        if member.image_path and os.path.exists(member.image_path):
            try:
                os.remove(member.image_path)
            except Exception as e:
                print(f"Error deleting image: {e}")

        db.session.delete(member)
        db.session.commit()

        # Full reload so removed member is gone
        face_recognizer.reload_known_faces()

        return jsonify({'success': True, 'message': f'Deleted {member.name}'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/get_family_members')
def get_family_members():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    members = FamilyMember.query.all()
    result = []
    for m in members:
        if m.image_path:
            if 'uploads/' in m.image_path:
                rel_path = m.image_path.split('uploads/')[-1]
            else:
                rel_path = os.path.basename(m.image_path)
            image_url = f'/uploads/{rel_path}'
        else:
            image_url = None

        result.append({
            'id': m.id,
            'name': m.name,
            'image_path': image_url,
            'full_path': m.image_path
        })
    return jsonify(result)


# ==============================================
# CAMERA ROUTES
# ==============================================
@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, camera_thread
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if not camera_active:
        camera_active = True
        live_status['camera_on'] = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.daemon = True
        camera_thread.start()
        return jsonify({'success': True, 'message': 'Camera started'})
    return jsonify({'success': False, 'message': 'Camera already running'})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    camera_active = False
    live_status['camera_on'] = False
    return jsonify({'success': True, 'message': 'Camera stopped'})


@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    def generate_frames():
        global frame_buffer
        while True:
            if frame_buffer is not None:
                ret, buffer = cv2.imencode('.jpg', frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def camera_loop():
    global camera_active, frame_buffer, alert_messages, live_status
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        camera_active = False
        live_status['camera_on'] = False
        return

    last_time = time.time()
    last_alert_time = 0
    ALERT_COOLDOWN = 30  # seconds between saving duplicate alerts

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)  # Mirror view

        # --- Face recognition ---
        face_names, unknown_detected = face_recognizer.process_frame(frame)
        face_locations = face_recognizer.get_face_positions()
        face_scores = face_recognizer.get_face_scores()

        # --- Stats ---
        known_count = sum(1 for n in face_names if n != "Unknown")
        unknown_count = sum(1 for n in face_names if n == "Unknown")
        live_status['total_faces'] = len(face_names)
        live_status['known_faces'] = known_count
        live_status['unknown_faces'] = unknown_count

        # --- FPS ---
        now = time.time()
        fps = 1.0 / max(now - last_time, 0.001)
        last_time = now
        live_status['fps'] = round(fps, 1)

        # --- Draw overlays ---
        draw_face_overlay(frame, face_locations, face_names, face_scores)
        draw_hud(frame, len(face_names), known_count, unknown_count, fps)

        # --- Alert on unknown (with cooldown) ---
        if unknown_detected and (now - last_alert_time) > ALERT_COOLDOWN:
            last_alert_time = now
            alert_message = f"Unknown person detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            with app.app_context():
                alert = Alert(alert_type='Unknown Person', message=alert_message)
                db.session.add(alert)
                db.session.commit()
            print(f"  [ALERT] Saved: {alert_message}")

        frame_buffer = frame.copy()
        time.sleep(0.03)

    cap.release()
    live_status['camera_on'] = False
    live_status['total_faces'] = 0
    live_status['known_faces'] = 0
    live_status['unknown_faces'] = 0
    print("Camera stopped")


# ==============================================
# ALERT ROUTES
# ==============================================
@app.route('/get_alerts')
def get_alerts():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    alerts = Alert.query.order_by(Alert.created_at.desc()).limit(10).all()
    result = [{'message': a.message, 'created_at': a.created_at.strftime('%Y-%m-%d %H:%M:%S')} for a in alerts]
    return jsonify(result)


# ==============================================
# WHATSAPP & EMAIL ROUTES
# ==============================================
@app.route('/send_whatsapp', methods=['POST'])
def send_whatsapp():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    latest_alert = Alert.query.order_by(Alert.created_at.desc()).first()
    if latest_alert:
        message = f"🚨 ALERT: {latest_alert.message}"
    else:
        message = WHATSAPP_MESSAGE

    encoded_message = urllib.parse.quote(message)
    whatsapp_url = f"https://wa.me/{WHATSAPP_NUMBER}?text={encoded_message}"

    return jsonify({
        'success': True,
        'whatsapp_url': whatsapp_url,
        'message': 'WhatsApp link generated'
    })


@app.route('/send_email', methods=['POST'])
def send_email():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    latest_alert = Alert.query.order_by(Alert.created_at.desc()).first()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if latest_alert:
        body = f"🚨 ALERT: {latest_alert.message}\n\nTimestamp: {timestamp}\n\nRegards,\nCCTV Monitoring System"
    else:
        body = EMAIL_BODY.format(timestamp=timestamp)

    encoded_subject = urllib.parse.quote(EMAIL_SUBJECT)
    encoded_body = urllib.parse.quote(body)
    email_url = f"mailto:{EMAIL_TO}?subject={encoded_subject}&body={encoded_body}"

    return jsonify({
        'success': True,
        'email_url': email_url,
        'message': 'Email link generated'
    })


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)