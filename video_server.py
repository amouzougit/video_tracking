from flask import Flask, render_template, Response, request, redirect, url_for
import os
import object_tracker as ot
import logging
import atexit

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def cleanup():
    if ot.cap is not None:
        ot.cap.stop()

atexit.register(cleanup)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            if not ot.set_video_source(filepath):
                return "Erreur lors de l'initialisation de la vidéo", 500
            return redirect(url_for('index'))
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {str(e)}")
            return "Erreur lors du téléchargement", 500

@app.route("/video_feed")
def video_feed():
    logger.info("Starting video feed")
    return Response(ot.streamVideo(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(host="localhost", port=5019, debug=True, threaded=True, use_reloader=False)