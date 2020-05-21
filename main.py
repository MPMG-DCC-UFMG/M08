#!/usr/bin/env python

from imageprocessor import ImageProcessor
from log import Log

import sys, os
import subprocess
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    files_path = sys.argv[1]
except:
    print('Processo precisa do argumento files_path com a localizacao dos arquivos')
    exit(1)

img_proc = ImageProcessor(files_path)
log_obj = Log(files_path)

# use_gpu, total_processes, child_conn
img_proc.process(True, 1, log_obj)    