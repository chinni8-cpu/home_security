# test_install.py
print("Testing installations...")

try:
    import flask
    print("✓ Flask installed")
except ImportError:
    print("✗ Flask not installed")

try:
    import flask_sqlalchemy
    print("✓ Flask-SQLAlchemy installed")
except ImportError:
    print("✗ Flask-SQLAlchemy not installed")

try:
    import cv2
    print("✓ OpenCV installed")
except ImportError:
    print("✗ OpenCV not installed")

try:
    import numpy
    print("✓ NumPy installed")
except ImportError:
    print("✗ NumPy not installed")

try:
    import pymysql
    print("✓ PyMySQL installed")
except ImportError:
    print("✗ PyMySQL not installed")

try:
    import dlib
    print("✓ dlib installed")
except ImportError:
    try:
        import mediapipe
        print("✓ MediaPipe installed (alternative to dlib)")
    except ImportError:
        print("✗ Neither dlib nor mediapipe installed")

print("\nAll tests complete!")