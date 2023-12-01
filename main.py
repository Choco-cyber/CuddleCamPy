import queue
import cv2
import mediapipe as mp
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import datetime
import os
import time
import shutil
import threading
from flask import Flask, send_file, Response, jsonify, render_template, request

message = 'Hello from Raspberry Pi'
message_queue = queue.Queue()

app = Flask(__name__)
capture = cv2.VideoCapture(0)

connected_mobile_app = None


# Object Detection function
def videoProcessing():
    global message
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    classNames = []
    classFile = 'cocoNames'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Visualize the detected poses
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Expression Detection
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_interpreter = tf.lite.Interpreter(model_path='emotion_detection_model_100epochs_no_opt.tflite')
    emotion_interpreter.allocate_tensors()
    emotion_input_details = emotion_interpreter.get_input_details()
    emotion_output_details = emotion_interpreter.get_output_details()
    emotion_input_shape = emotion_input_details[0]['shape']
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Variables for fall detection
    fall_count = 0
    fall_threshold = 1  # Number of consecutive frames to detect a fall
    is_fallen = False
    sad_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            success, img = capture.read()

            # Object Detection
            classIds, confs, bbox = net.detect(img, confThreshold=0.55)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if len(classNames) > 0 and classId - 1 < len(classNames):
                        # cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                        if classId == 17 or classId == 18:
                            message_queue.put("There is an animal in the room")
                            print("There is an animal in the room")

                        # Identify the person
                        if classId == 1:
                            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            image.flags.writeable = False
                            results = pose.process(image)  # Make detection
                            image.flags.writeable = True  # Recolor back to BGR
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            # Fall detection logic
                            if results.pose_landmarks is not None:
                                landmarks = results.pose_landmarks.landmark

                                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                                hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

                                if left_ankle < hip and right_ankle < hip:
                                    if not is_fallen:
                                        fall_count += 1
                                        if fall_count >= fall_threshold:
                                            message_queue.put("Fall detected!")
                                            print("Fall detected!")

                                        is_fallen = True
                                else:
                                    is_fallen = False
                                    fall_count = 0

                                # Render detection
                                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                                 circle_radius=2),
                                                          mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2,
                                                                                 circle_radius=2))

                                # Expression Detection
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                faces = face_classifier.detectMultiScale(gray, 1.1, 3, 5)
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    roi_gray = gray[y:y + h, x:x + w]
                                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                                    roi = roi_gray.astype('float') / 255.0
                                    roi = img_to_array(roi)
                                    roi = np.expand_dims(roi, axis=0)

                                    emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
                                    emotion_interpreter.invoke()
                                    emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

                                    emotion_label = class_labels[emotion_preds.argmax()]
                                    emotion_label_position = (x, y)
                                    cv2.putText(img, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)

                                    if emotion_label == 'Sad':
                                        sad_count += 1
                                        if sad_count > 25:
                                            message_queue.put("Not in a good mood")
                                            print("Not in a good mood")
                                            sad_count = 0

                            # Display the processed output
                            cv2.imshow('Processed output', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()


def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        if not ret:
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)

        # Yield the output frame as a byte array
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/get_video', methods=['GET'])
def get_latest_video():
    latest_video = get_latest_completed_video()
    if latest_video:
        folder_path = 'D:\\Assignments\\NLP\\cuddlecam\\video'
        source_path = os.path.join(folder_path, latest_video)
        return send_file(source_path, as_attachment=True)
    else:
        return "No completed video files found in the folder."


def record_and_cleanup_videos():
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_folder = 'D:\\Assignments\\NLP\\cuddlecam\\video'  # Folder to store video segments
    os.makedirs(output_folder, exist_ok=True)

    # Variables for segmentation and cleanup
    segment_duration_seconds = 60  # 1 minute in seconds
    cleanup_duration_seconds = 180  # 3 minutes in seconds
    segment_start_time = time.time()
    cleanup_start_time = time.time()
    output = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        current_time = time.time()

        # Create a new segment if needed
        if current_time - segment_start_time >= segment_duration_seconds:
            # Save and close the current segment
            if output is not None:
                output.release()

            # Create a new segment with a timestamp in the filename
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_path = os.path.join(output_folder, f'recording_{timestamp}.avi')
            output = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

            # Update the segment start time
            segment_start_time = current_time

        # Write the frame to the current segment
        if output is not None:
            output.write(frame)

        # Perform cleanup
        if current_time - cleanup_start_time >= cleanup_duration_seconds:
            files = os.listdir(output_folder)
            files.sort()  # Sort files by name (timestamp)

            # Keep the last saved video and the one being recorded
            files_to_delete = files[:-2]
            for file_name in files_to_delete:
                file_path = os.path.join(output_folder, file_name)
                os.remove(file_path)

            cleanup_start_time = current_time

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    capture.release()
    if output is not None:
        output.release()
    cv2.destroyAllWindows()


def get_latest_completed_video():
    folder_path = 'D:\\Assignments\\NLP\\cuddlecam\\video'
    expected_duration = 60  # Expected duration in seconds

    try:
        files = os.listdir(folder_path)
        completed_videos = []

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            file_duration = os.path.getmtime(file_path) - os.path.getctime(file_path)

            # Check if the video is completed based on expected duration
            if abs(file_duration - expected_duration) < 10:
                completed_videos.append((file_name, os.path.getmtime(file_path)))

        if completed_videos:
            # Sort completed videos by modification time (latest first)
            completed_videos.sort(key=lambda x: x[1], reverse=True)
            return completed_videos[0][0]
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None


def copy_latest_completed_video_to_desktop():
    latest_video = get_latest_completed_video()
    if latest_video:
        folder_path = 'D:\\Assignments\\NLP\\cuddlecam\\video'
        desktop_path = os.path.expanduser("~/Desktop")

        source_path = os.path.join(folder_path, latest_video)
        destination_path = os.path.join(desktop_path, latest_video)

        try:
            shutil.copyfile(source_path, destination_path)
            print(f'Copied {latest_video} to the desktop.')
        except Exception as e:
            print(f'An error occurred: {e}')
    else:
        print("No completed video files found in the folder.")


def user_input_thread():
    while True:
        user_input = input("Do you want to copy the latest completed video to the desktop? (yes/no): ")
        if user_input.lower() == "yes":
            copy_latest_completed_video_to_desktop()


# Displaying webpage
@app.route('/')
def index():
    return render_template('index.html')


# Displaying the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Connecting to the mobile application
@app.route('/connect', methods=['GET'])
def connect():
    global connected_mobile_app
    connected_mobile_app = request.remote_addr
    return 'Connected successfully'


# Disconnecting to the mobile application
@app.route('/disconnect', methods=['GET'])
def disconnect():
    global connected_mobile_app
    connected_mobile_app = None
    return 'Disconnected successfully'


# Displaying the current connectivity status
@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    if connected_mobile_app is not None:
        return jsonify({'status': 'connected'})
    else:
        return jsonify({'status': 'disconnected'})


@app.route('/share_message', methods=['GET'])
def share_message():
    if connected_mobile_app is not None:
        return jsonify({'message': message_queue.get()})
    else:
        return jsonify({'message': 'Stop'})


# Running the threads simultaneously
if __name__ == "__main__":
    # Start the video recording and cleanup thread
    video_thread = threading.Thread(target=record_and_cleanup_videos)
    video_thread.daemon = True
    video_thread.start()

    # Start the user input thread
    user_input_thread = threading.Thread(target=user_input_thread)
    user_input_thread.daemon = True
    user_input_thread.start()

    processingThread = threading.Thread(target=videoProcessing)
    processingThread.daemon = True
    processingThread.start()

    flask_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 3000})
    flask_thread.daemon = True
    flask_thread.start()

    # Keep the main thread alive to allow both threads to run indefinitely
    while True:
        time.sleep(1)
