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
from report import Report
from log import Log

from flask import Blueprint, render_template, request, jsonify,\
                    flash, redirect, url_for
from . import db
from . import dialog
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

#### Global variables
log_obj = Log()
file_searcher = None
id_process = None
global_path = 'Nome do Diretório'
####

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
    file_searcher = FileSearcher(local_path)
    file_searcher.get_from_directory(log_obj, 0, verbose_fs=True) #signal_msg, task_id
    print(log_obj.buffer)

    return jsonify(path=global_path)

@main.route('/new_analysis', methods=['POST', 'GET'])
@login_required
def new_analysis():
    if request.method == 'POST':
        print(request.form['info3'])
    return render_template('new_analysis.html', name=current_user.name, path=global_path)

@main.route('/idprocess', methods=['POST'])
@login_required
def IDset():
    global id_process
    global log_obj
    
    id_process = request.form.get('id-process')
    
    if id_process in log_obj.all_logs: 
        flash('O identificador {} já existe.'.format(id_process))
        return redirect(url_for('main.new_analysis')) 
    
    log_obj.set_id(id_process)
    log_obj.send(('imprime', 'identificador definido com sucesso'))
    return '', 204

@main.route('/IMGprocessor', methods=['POST', 'GET'])
@login_required
def IMGprocessor():
    global file_searcher
    global log_obj
    
    if id_process is None: 
        flash('Defina o identificador da análise.'.format(id_process))
        return redirect(url_for('main.new_analysis')) 
    
    img_proc = ImageProcessor(file_searcher.files["images"])
    img_proc.process(True, 1, log_obj)
    return '', 204

@main.route('/VIDprocessor')
@login_required
def VIDprocessor():
    global file_searcher
    global log_obj

    if id_process is None: 
        flash('Defina o identificador da análise.'.format(id_process))
        return redirect(url_for('main.new_analysis')) 
    
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/IMGVIDprocessor')
@login_required
def IMGVIDprocessor():
    global file_searcher
    global log_obj
    
    if id_process is None: 
        flash('Defina o identificador da análise.'.format(id_process))
        return redirect(url_for('main.new_analysis')) 
    
    img_proc = ImageProcessor(file_searcher.files["images"])
    img_proc.process(True, 1, log_obj)
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/IMGreport', methods=['POST', 'GET'])
@login_required
def IMGreport():
    global file_searcher
    global log_obj
    
    img_report = Report(log_obj.log_path, log_obj.logfile) 
    html_path = img_report.generate_img(return_path=True)
    
    return render_template('report.html', name=current_user.name,) 
    
@main.route('/VIDreport')
@login_required
def VIDreport():
    global file_searcher
    global log_obj

    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/IMGVIDreport')
@login_required
def IMGVIDreport():
    global file_searcher
    global log_obj

    img_proc = ImageProcessor(file_searcher.files["images"])
    img_proc.process(True, 1, log_obj)
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/report', methods=['POST', 'GET'])
@login_required
def report():
    return render_template('report.html', name=current_user.name, id_report="123456", path="teste/teste/teste")

@main.route('/search_analysis', methods=['POST', 'GET'])
@login_required
def search_analysis():
    return render_template('search_analysis.html')
