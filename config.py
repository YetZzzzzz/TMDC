# *_*coding:utf-8 *_*
import os
import sys
import socket

import torch

## gain linux ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',8080))
        ip= s.getsockname()[0]
    finally:
        s.close()
    return ip

############ For LINUX ##############
# path
DATA_DIR = {
	'CMUMOSI': './gcnet_datasets',   # for nlpr
	'CMUMOSEI': './gcnet_datasets',# for nlpr
	'IEMOCAPSix': './gcnet_datasets', # for nlpr
	'IEMOCAPFour': './gcnet_datasets', # for nlpr
}
PATH_TO_RAW_AUDIO = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subaudio'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
}
PATH_TO_RAW_FACE = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subvideofaces'), # without openfac
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideofaces'),
}
PATH_TO_TRANSCRIPTIONS = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription.csv'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription.csv'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'transcription.csv'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'cmu_mosi_features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'cmu_mosei_features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'iemocap_features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'iemocap_features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}

