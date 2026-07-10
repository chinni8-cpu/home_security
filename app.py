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
# FACE RECOGNIZER CLASS WITH RELOAD FUNCTION
# ==============================================
class SimpleFaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_names = []
        self.load_known_faces()

    def get_face_features(self, face_img):
        """Extract face features using LBP (Local Binary Patterns)"""
        face_resized = cv2.resize(face_img, (100, 100))
        features = face_resized.flatten()
        features = features / 255.0
        return features

    def load_known_faces(self):
        """Load all family members from database"""
        with app.app_context():
            family_members = FamilyMember.query.all()
            
        print(f"🔄 Loading {len(family_members)} family members...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        for member in family_members:
            if member.image_path and os.path.exists(member.image_path):
                img = cv2.imread(member.image_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, 1.1, 5, minSize=(50, 50)
                    )
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        features = self.get_face_features(face_roi)
                        self.known_face_encodings.append(features)
                        self.known_face_names.append(member.name)
                        print(f"✅ Loaded: {member.name}")
                    else:
                        print(f"⚠️ No face detected in image: {member.name}")
            else:
                print(f"⚠️ File not found: {member.image_path}")
        
        print(f"✅ Total loaded: {len(self.known_face_names)} members")

    def reload_known_faces(self):
        """Reload all family members from database (clear and reload)"""
        self.load_known_faces()
        print(f"🔄 Reloaded {len(self.known_face_names)} family members")

    def add_family_member(self, name, image_path):
        """Add a new family member"""
        img = cv2.imread(image_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(50, 50)
            )
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                features = self.get_face_features(face_roi)
                self.known_face_encodings.append(features)
                self.known_face_names.append(name)
                print(f"➕ Added: {name}")
                return True
        return False

    def process_frame(self, frame):
        """Process frame and detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60, 60)
        )
        
        self.face_locations = []
        self.face_names = []
        unknown_detected = False
        
        if len(self.known_face_encodings) == 0:
            for (x, y, w, h) in faces:
                self.face_locations.append((y, x+w, y+h, x))
                self.face_names.append("No members registered")
            return self.face_names, False
        
        for (x, y, w, h) in faces:
            self.face_locations.append((y, x+w, y+h, x))
            face_roi = gray[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                self.face_names.append("Unknown")
                continue
            
            try:
                detected_features = self.get_face_features(face_roi)
            except:
                self.face_names.append("Unknown")
                continue
            
            best_match_name = "Unknown"
            best_score = 0.0
            
            for i, known_features in enumerate(self.known_face_encodings):
                similarity = self.cosine_similarity(detected_features, known_features)
                diff = np.linalg.norm(detected_features - known_features)
                score = similarity * 0.7 + (1 / (1 + diff)) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match_name = self.known_face_names[i]
            
            if best_score > 0.60:
                name = best_match_name
                print(f"✅ Recognized: {name} (Score: {best_score:.2f})")
            else:
                name = "Unknown"
                unknown_detected = True
                print(f"❌ Unknown (Score: {best_score:.2f})")
            
            self.face_names.append(name)
        
        return self.face_names, unknown_detected

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_face_positions(self):
        return self.face_locations

# Initialize face recognizer
face_recognizer = SimpleFaceRecognizer()

# Camera control
camera_active = False
camera_thread = None
frame_buffer = None
alert_messages = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
    """Check what files exist in uploads folder"""
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
    """Debug face recognition"""
    result = {
        'known_faces': len(face_recognizer.known_face_names),
        'known_names': face_recognizer.known_face_names,
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
                    faces = face_recognizer.face_cascade.detectMultiScale(gray, 1.1, 5)
                    result[f'{member.name}_faces_detected'] = len(faces)
                else:
                    result[f'{member.name}_error'] = 'Could not load image'
            else:
                result[f'{member.name}_error'] = 'File not found'
    
    return jsonify(result)

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
        
        face_recognizer.add_family_member(name, filepath)
        
        return jsonify({'success': True, 'message': 'Family member added successfully'})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/delete_family_member/<int:member_id>', methods=['DELETE'])
def delete_family_member(member_id):
    """Delete a family member and reload face recognition"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        member = FamilyMember.query.get(member_id)
        if not member:
            return jsonify({'error': 'Member not found'}), 404
        
        # Delete the image file if it exists
        if member.image_path and os.path.exists(member.image_path):
            try:
                os.remove(member.image_path)
                print(f"🗑️ Deleted image: {member.image_path}")
            except Exception as e:
                print(f"Error deleting image: {e}")
        
        # Delete from database
        db.session.delete(member)
        db.session.commit()
        
        # Reload face recognizer
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
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    def generate_frames():
        global frame_buffer
        while True:
            if frame_buffer is not None:
                ret, buffer = cv2.imencode('.jpg', frame_buffer)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def camera_loop():
    global camera_active, frame_buffer, alert_messages
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        camera_active = False
        return
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        face_names, unknown_detected = face_recognizer.process_frame(frame)
        face_locations = face_recognizer.get_face_positions()
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            if name == "Unknown" and unknown_detected:
                alert_message = f"Unknown person detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if alert_message not in alert_messages:
                    alert_messages.append(alert_message)
                    with app.app_context():
                        alert = Alert(alert_type='Unknown Person', message=alert_message)
                        db.session.add(alert)
                        db.session.commit()
        
        frame_buffer = frame.copy()
        time.sleep(0.03)
    
    cap.release()
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
    """Generate WhatsApp link with pre-filled message"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get latest alert message if available
    latest_alert = Alert.query.order_by(Alert.created_at.desc()).first()
    if latest_alert:
        message = f"🚨 ALERT: {latest_alert.message}"
    else:
        message = WHATSAPP_MESSAGE
    
    # Encode message for URL
    encoded_message = urllib.parse.quote(message)
    
    # Create WhatsApp URL
    whatsapp_url = f"https://wa.me/{WHATSAPP_NUMBER}?text={encoded_message}"
    
    return jsonify({
        'success': True, 
        'whatsapp_url': whatsapp_url,
        'message': 'WhatsApp link generated'
    })

@app.route('/send_email', methods=['POST'])
def send_email():
    """Generate email link with pre-filled subject and body"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get latest alert
    latest_alert = Alert.query.order_by(Alert.created_at.desc()).first()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if latest_alert:
        body = f"🚨 ALERT: {latest_alert.message}\n\nTimestamp: {timestamp}\n\nRegards,\nCCTV Monitoring System"
    else:
        body = EMAIL_BODY.format(timestamp=timestamp)
    
    # Encode for URL
    encoded_subject = urllib.parse.quote(EMAIL_SUBJECT)
    encoded_body = urllib.parse.quote(body)
    
    # Create mailto link
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