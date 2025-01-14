import os
import random
import numpy as np
import cv2
import threading
from ultralytics import YOLO
from videoasync import VideoCaptureAsync
import tracker as hasher

lock = threading.Lock()
model_path = "./models/yolov8n.pt"
model = YOLO(model_path)
object_ids = {}
hasher_object = hasher.ObjectHasher()

def get_color_table(class_num, seed=50):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

colortable = get_color_table(80)
cap = None

def set_video_source(filepath):
    global cap
    if cap is not None:
        cap.stop()
    cap = VideoCaptureAsync(src=filepath).start()

def track_object(model, image_np):
    print("------- Frame Processing -------")
    try:
        results = model(image_np)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf > 0.5:
                    print(f"Détection: classe={model.names[class_id]}, conf={conf:.2f}, box=({x1},{y1},{x2},{y2})")
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), colortable[class_id], 3)
                    label = f"{model.names[class_id]}: {conf:.2f}"
                    cv2.putText(image_np, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colortable[class_id], 2)
        
        return image_np
    except Exception as e:
        print(f"Erreur dans track_object: {str(e)}")
        return image_np


def set_video_source(filepath):
    global cap
    print(f"Initialisation de la vidéo: {filepath}")
    if cap is not None:
        cap.stop()
    try:
        cap = VideoCaptureAsync(src=filepath)
        if not cap.start():
            print("Erreur: Impossible de démarrer la capture vidéo")
            return False
        print("Capture vidéo initialisée avec succès")
        return True
    except Exception as e:
        print(f"Erreur d'initialisation de la vidéo: {str(e)}")
        return False

def streamVideo():
    global lock, cap
    try:
        while True:
            if cap is None:
                print("Capture vidéo non initialisée")
                break

            retrieved, frame = cap.read()
            if not retrieved or frame is None:
                print("Redémarrage de la vidéo...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Retour au début
                continue

            with lock:
                processed_frame = track_object(model, frame)
                if processed_frame is None:
                    continue

            success, buffer = cv2.imencode('.jpg', processed_frame)
            if not success:
                print("Erreur d'encodage du frame")
                continue

            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + 
                  buffer.tobytes() + b'\r\n')

    except Exception as e:
        print(f"Erreur dans streamVideo: {str(e)}")
    finally:
        print("Nettoyage des ressources vidéo")
        if cap is not None:
            cap.stop()