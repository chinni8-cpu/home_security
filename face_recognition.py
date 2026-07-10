import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO
import json

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.last_known_name = None
        self.process_this_frame = True
        self.load_known_faces()

    def load_known_faces(self):
        """Load all known faces from the database"""
        from database import FamilyMember
        from app import app
        
        with app.app_context():
            family_members = FamilyMember.query.all()
            
        for member in family_members:
            if member.image_path and os.path.exists(member.image_path):
                image = face_recognition.load_image_file(member.image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    self.known_face_names.append(member.name)

    def add_family_member(self, name, image_path):
        """Add a new family member"""
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            self.known_face_encodings.append(encoding[0])
            self.known_face_names.append(name)
            return True
        return False

    def process_frame(self, frame):
        """Process a frame and detect faces"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        unknown_detected = False
        
        for face_encoding in self.face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # If match found, get the name
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                self.last_known_name = name
            else:
                if self.last_known_name is not None:
                    name = self.last_known_name
                else:
                    unknown_detected = True

            self.face_names.append(name)

        return self.face_names, unknown_detected

    def get_face_positions(self):
        """Get face positions for drawing rectangles"""
        return self.face_locations

    def encode_image_to_base64(self, image):
        """Convert image to base64 for web display"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')