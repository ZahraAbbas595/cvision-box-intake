import os

CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.45'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.30'))
SERVICE_VERSION = '0.1.0'
MODEL_NAME = 'carton-yolov8n'
MODEL_VERSION = 'ft-v1'
SCHEMA_VERSION = '1.0'
