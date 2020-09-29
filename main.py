#!/usr/bin/env python
import sys, os
import subprocess
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('./M08')
from imageprocessor import ImageProcessor
from videoprocessor import VideoProcessor
from filesearcher import FileSearcher
from log import Log

from flask import Blueprint, render_template, request, jsonify
from . import db
from . import dialog
from flask_login import login_required, current_user

main = Blueprint('main', __name__)
log_obj = None
file_searcher = None

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/home')
@login_required
def home():
    return render_template('home.html')

@main.route('/SOdialog')
@login_required
def SOdialog():
    global file_searcher
    global log_obj
    global global_path
    
    local_path = dialog._open_dialog_file()
    global_path = local_path
    log_obj = Log(local_path)
    file_searcher = FileSearcher(local_path)
    file_searcher.get_from_directory(log_obj, 0, verbose_fs=True) #signal_msg, task_id
    print(log_obj.buffer)
    
    return jsonify(path=global_path)

@main.route('/new_analysis', methods=['POST', 'GET'])
@login_required
def new_analysis():
    if request.method == 'POST':
        print(request.form['info3'])
    return render_template('new_analysis.html', name=current_user.name)

@main.route('/IMGprocessor', methods=['POST', 'GET'])
@login_required
def IMGprocessor():
    global file_searcher
    global log_obj
    
    print('image processor')
    img_proc = ImageProcessor(file_searcher.files["images"])
    print('\n\nprocess\n\n')
    img_proc.process(True, 1, log_obj)
    return '', 204

@main.route('/VIDprocessor')
@login_required
def VIDprocessor():
    global file_searcher
    global log_obj
    
    print('-'*25+'\nIniciando analise de vídeos')
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    print('\nConcluído.'+'\n'+'-'*25)
    return '', 204

@main.route('/IMGVIDprocessor')
@login_required
def IMGVIDprocessor():
    global file_searcher
    global log_obj
    
    img_proc = ImageProcessor(file_searcher.files["images"])
    img_proc.process(True, 1, log_obj)
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/search_analysis', methods=['POST', 'GET'])
@login_required
def search_analysis():
    return render_template('search_analysis.html')
