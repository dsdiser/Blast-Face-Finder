import shutil
from alive_progress import alive_bar
import face_recognition
import os, sys
import cv2
import numpy as np
import math


# Helpers
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def ms_to_hours(millis):
    # convert ms to zero padded hours minutes and seconds
    seconds, milliseconds = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d-%02d-%02d-%03d" % (hours, minutes, seconds, milliseconds)


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        for video in os.listdir('video'):
            filename_no_ext = os.path.splitext(video)[0]
            video_capture = cv2.VideoCapture(f"video/{video}")

            if not video_capture.isOpened():
                sys.exit('Video source not found...')
            # clear and then create folder for outputting matching frames
            # if os.path.exists(f'output/{filename_no_ext}'):
            #     shutil.rmtree(f'output/{filename_no_ext}')
            # os.makedirs(f'output/{filename_no_ext}')

            # start at a specific frame number
            starting_frame = 197609*4 - 1
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
            frames_to_process = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 4 - 1
            prev_face_count = 0
            with alive_bar(frames_to_process, title=filename_no_ext) as bar:
                bar(starting_frame//4, skipped=True)
                while video_capture.isOpened():
                    ret, frame = video_capture.read()

                    if not ret:
                        break

                    # Only process every 4th frame of video to save time
                    if video_capture.get(cv2.CAP_PROP_POS_FRAMES) % 4 != 0:
                        continue

                    # Resize frame of video to 1/2 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Find all the faces and face encodings in the current frame of video
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                    self.face_names = []
                    known_face_count = 0
                    for face_encoding in self.face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=.4)
                        # Calculate the shortest distance to face
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            known_face_count += 1
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                            self.face_names.append(f'{name} ({confidence})')
                        else:
                            self.face_names.append('')

                    # avoids unnecessary images by limiting our save to newly seeing a face or no longer seeing a face
                    if (not prev_face_count and known_face_count) or (prev_face_count and not known_face_count):
                        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                            if name:
                                # Create the frame with the name
                                # cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                cv2.putText(small_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                        # Save the resulting image
                        timestamp_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                        img_path = os.path.join(os.getcwd(), 'output', filename_no_ext, f'{ms_to_hours(timestamp_ms)}-{known_face_count}.png')
                        cv2.imwrite(img_path, small_frame)
                    prev_face_count = known_face_count
                    # update progress bar
                    bar()
            # Release handle to the video
            video_capture.release()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
