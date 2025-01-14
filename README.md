# Video Object Tracking

Ce projet utilise YOLOv8 pour détecter et suivre des objets dans des vidéos. Il permet de télécharger une vidéo, de détecter des objets dans chaque frame et de suivre ces objets au fil du temps.

## Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation

1. Clonez le dépôt :
   ```sh
   git clone https://github.com/amouzougit/video_tracking.git
   cd video_tracking

2. Créez un environnement virtuel et activez-le :
python -m venv env
source env/bin/activate  # Sur Windows, utilisez `env\Scripts\activate`

3. Installez les dépendances :

pip install -r requirements.txt

4. Téléchargez le modèle YOLOv8 et placez-le dans le répertoire models :

mkdir models
mv yolov8n.pt models/

Utilisation


Lancez le serveur Flask :
python video_server.py

Ouvrez votre navigateur et accédez à http://localhost:5019.

Téléchargez une vidéo via l'interface web.

Regardez le flux vidéo avec les objets détectés et suivis.


