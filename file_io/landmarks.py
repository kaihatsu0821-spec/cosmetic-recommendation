#landmarks.py
import cv2
import mediapipe as mp

detector = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)


def get_landmarks(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    return results.multi_face_landmarks[0]