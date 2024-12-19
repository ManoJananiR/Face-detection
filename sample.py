import cv2
import numpy as np
from deepface import DeepFace

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def detect_person_in_video(image_path, video_path, output_path):
    # Load the known face image and extract its embedding
    known_image = cv2.imread(image_path)
    known_face_embedding = DeepFace.represent(known_image, model_name='VGG-Face', enforce_detection=True)[0]['embedding']

    # Load the video
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    frame_count = 0
    detection_interval = 10  # Perform detection every 10 frames to reduce computation

    # Variables to store the last recognized face location, age, and emotion
    last_bbox = None
    last_age = "Unknown"
    last_emotion = "Unknown"
    stable_display_counter = 0
    stable_display_limit = 30  # Number of frames to keep showing last detection

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % detection_interval == 0:
            try:
                # Extract faces from the current video frame
                detected_faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)

                for face in detected_faces:
                    # Extract face region and face embedding
                    x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                    detected_face = frame[y:y+h, x:x+w]

                    # Calculate the embedding for the detected face
                    detected_face_embedding = DeepFace.represent(detected_face, model_name='VGG-Face', enforce_detection=False)[0]['embedding']

                    # Compute cosine similarity between known face and detected face
                    similarity = cosine_similarity(known_face_embedding, detected_face_embedding)

                    # If similarity is above a threshold, it's a match (detect only for known face)
                    if similarity > 0.65:
                        # Store the bounding box and detected age/emotion
                        last_bbox = (x, y, w, h)
                        analysis = DeepFace.analyze(detected_face, actions=['age', 'emotion'], enforce_detection=False)
                        last_age = analysis[0].get('age', 'Unknown')
                        last_emotion = analysis[0].get('dominant_emotion', 'Unknown')
                        stable_display_counter = 0  # Reset the counter when a new detection happens
                        break

            except Exception as e:
                print(f"Error in processing frame: {e}")

        # If we have detected a known face recently or just now
        if last_bbox and stable_display_counter < stable_display_limit:
            x, y, w, h = last_bbox

            # Draw the bounding box around the last detected known face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw age and emotion below the bounding box
            font_scale = 2.2  # Increased font scale for larger text
            text_color = (0, 0, 255)  # Text color (BGR)
            thickness = 5  # Thickness of the text

            age_text = f'Age: {last_age}'
            emotion_text = f'Emotion: {last_emotion}'
            age_text_size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            emotion_text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

            # Draw the age text below the bounding box
            cv2.putText(frame, age_text, (x, y + h + age_text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            # Draw the emotion text below the age text
            cv2.putText(frame, emotion_text, (x, y + h + age_text_size[1] + emotion_text_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            # Increment the counter for how long we should keep showing the last detected face
            stable_display_counter += 1

        else:
            # If no known face detected for a while, we keep writing the frame without detection info
            stable_display_counter += 1

        # Write the modified frame to the output video
        out.write(frame)
        frame_count += 1

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
detect_person_in_video('knownn.png', 'clip1.mp4', 'output_videof.avi')
