# CCTV Monitoring System

A Flask-based CCTV monitoring application with user authentication, family member registration, live camera feed, and alert generation for unknown persons.

## Features

- User registration and login
- Family member image upload and storage
- Face detection and recognition for known family members
- Live video stream from the webcam
- Alerts for unknown persons
- WhatsApp and email alert links

## Project Structure

- app.py - Main Flask application
- database.py - Database helpers (if present)
- face_recognition.py - Face recognition logic
- templates/ - HTML templates for login, registration, and dashboard
- static/ - CSS and JavaScript assets
- uploads/family_members/ - Stored family member images

## Requirements

Python 3.8+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or on Windows, you can run:

```bat
install_all.bat
```

## Database Setup

This project uses MySQL with the following default configuration in app.py:

- Database: cctv_db
- Username: root
- Password: empty
- Host: localhost

Make sure MySQL is installed and running, then create the database:

```sql
CREATE DATABASE cctv_db;
```

If needed, update the database connection string in app.py.

## Running the Application

Start the Flask app:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Usage

1. Register a new account
2. Log in to the dashboard
3. Add family members by uploading clear face images
4. Start the camera to monitor live video
5. Review alerts for unknown persons

## Notes

- A working webcam is required for live monitoring
- Face recognition accuracy depends on image quality and lighting
- The system may need tuning for better detection results in your environment

## License

This project is for educational and demonstration purposes.
