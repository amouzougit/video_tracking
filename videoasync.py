import threading
import cv2

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()


    def set(self, key, value):
        self.cap.set(key, value)

    def start(self):
        if self.started:
            print('[Warning] Asynchronous video capturing is already started.')
            return  None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return True
    
    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame
    
    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()



# # Model preparation
    def load_model(self, model_path):
        model_dir = pathlib.Path(model_path) / "saved_model"
        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']
        return model
    
    def some_method(self):
        model_path = "/model/yolov8n.pt" 
        model = self.load_model(model_path)
        # Utilisez le modèle ici

# Function to implement infinite while loop to read video frames and generate the output for web browser
def streamVideo():
    global lock, cap, model
    while True:
        retrieved, frame = cap.read()
        if retrieved:
            with lock:
                frame = track_object(model, frame)
            
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.stop()
            break
    
    # When everything done, release the capture
    cap.stop()
    cv2.destroyAllWindows()


# Function to track the object in the video frame
def getCropped(image_np, xmin, ymin, xmax, ymax):
    return image_np[ymin:ymax, xmin:xmax]

def resize(cropped_image, size=8):
    resized = cv2.resize(cropped_image, (size+1, size))
    return resized

def getHash(resized_image):
    diff = resized_image[:, 1:] > resized_image[:, :-1]
    # convert the difference image to a hash
    dhash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    return int(np.array(dhash, dtype="float64"))