import os
from keras.models import load_model
from flask import Flask
import eventlet
import socketio
import base64
from io import BytesIO
from PIL import Image
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sio = socketio.Server()
# '__main__'
app = Flask(__name__)

maxSpeed = 5


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocessing(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed

    print(f'steering: {steering} - speed: {speed}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    print("Run...")

    print("Load model...")
    model_path = os.path.join(c.model_dir, '0.0001-lr10-e1000-spe20-ed.h5')
    model = load_model(model_path)
    app = socketio.Middleware(sio, app)

    # LISTEN TO PORT 4567
    print("Listen to port 4567...")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
