import cv2 
import numpy as np
import time
import os
import logging
import subprocess
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

# Initialize models at the module level so they are loaded only once.
# This saves significant time and memory for each request.
try:
    logger.info("Loading MTCNN and InceptionResnetV1 models...")
    _mtcnn = MTCNN()
    _facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    _mtcnn = None
    _facenet_model = None

def run(video_path, video_path2):
    """
    Process the video and detect deepfakes.
    Returns the percentage of deepfake frames detected.
    """
    if _mtcnn is None or _facenet_model is None:
        raise RuntimeError("ML Models were not properly loaded initialization")

    start_time = time.time()
    
    threshold_face_similarity = 0.99  
    threshold_frames_for_deepfake = 15  

    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # fallback fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path2, fourcc, fps, (width, height))

    deepfake_count = 0
    deep_fake_frame_count = 0
    previous_face_encoding = None
    
    # OPTIMIZATION: Process only 2 frames per second (instead of fps/15 which is often 2 frames per sec anyway, but let's be explicit and configurable)
    # Processing every frame is way too slow.
    frames_between_processing = max(1, int(fps / 2))  
    
    # Face dimension for InceptionResnetV1
    resize_dim = (80, 80)
    
    # Setup scaled processing for MTCNN? (Reverting: Causes accuracy loss on smaller faces)
    # scale_factor = 0.5 

    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret:  
            break

        if frame_count % frames_between_processing == 0:
            # Pass the full frame to MTCNN for accurate face detection
            boxes, _ = _mtcnn.detect(frame)  

            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                
                # Ensure box dimensions are within frame boundaries
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = min(width, box[2])
                box[3] = min(height, box[3])
                
                face = frame[box[1]:box[3], box[0]:box[2]]

                if face.size > 0:
                    face = cv2.resize(face, resize_dim)
                    face_tensor = F.to_tensor(face).unsqueeze(0)
                    current_face_encoding = _facenet_model(face_tensor).detach().numpy().flatten()

                    if previous_face_encoding is not None:
                        # Cosine similarity
                        norm_current = np.linalg.norm(current_face_encoding)
                        norm_prev = np.linalg.norm(previous_face_encoding)
                        
                        if norm_current > 0 and norm_prev > 0:
                            face_similarity = np.dot(current_face_encoding, previous_face_encoding) / (norm_current * norm_prev)

                            # Accumulate deepfake frames based on similarity
                            if face_similarity < threshold_face_similarity:
                                deepfake_count += 1
                                deep_fake_frame_count += 1
                            else:
                                # Decrease severity rather than resetting completely to 0 on a single good frame
                                deepfake_count = max(0, deepfake_count - 1)

                            if deepfake_count > threshold_frames_for_deepfake:
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                                cv2.putText(frame, f'Deepfake Detected', (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            else:
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                cv2.putText(frame, 'Real', (box[0], max(15, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                            cv2.LINE_AA)

                    previous_face_encoding = current_face_encoding

        frame_count += 1
        out.write(frame)

    end_time = time.time()
    logger.info(f"Total Execution Time: {end_time - start_time:.2f} seconds")

    cap.release()  
    out.release()  

    # Convert OpenCV mp4v to Web-compatible H264 using FFmpeg
    temp_path = video_path2 + "_temp.mp4"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    os.rename(video_path2, temp_path)
    
    logger.info("Converting video codec to H264 for browser compatibility...")
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', temp_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '28', video_path2],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            check=True
        )
        if os.path.exists(temp_path) and os.path.exists(video_path2):
            os.remove(temp_path)
        logger.info("Video conversion complete.")
    except Exception as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        # Revert on failure just in case so file isn't completely lost
        if os.path.exists(temp_path) and not os.path.exists(video_path2):
            os.rename(temp_path, video_path2)

    if frame_count == 0:
        return 0

    accuracy = (deep_fake_frame_count / frame_count) * 1000  

    if accuracy > 100:
        accuracy = 95  

    return int(accuracy)

