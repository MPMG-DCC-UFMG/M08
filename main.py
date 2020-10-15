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
id_process = ''
global_path = 'Caminho do Diretório'
####

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/home')
@login_required
def home():
    return render_template('home.html')

@main.route('/log', methods=['POST', 'GET'])
@login_required
def funcao_teste():
#     flash(log_obj.buffer, 'log')
    return redirect(url_for('main.new_analysis')) 


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

    return jsonify(path=global_path)

@main.route('/new_analysis', methods=['POST', 'GET'])
@login_required
def new_analysis():
    if request.method == 'POST':
        print(request.form['info3'])
    return render_template('new_analysis.html', name=current_user.name, path=global_path, 
                                                id_process=id_process, buffer=log_obj.buffer)

@main.route('/idprocess', methods=['GET','POST'])
@login_required
def IDset():
    global id_process
    global log_obj
    print('\nIDset\n')
    id_process = request.form.get('id-process')
    
    if id_process in log_obj.all_logs: 
        flash('O identificador {} já existe.'.format(id_process), 'error')
        return redirect(url_for('main.new_analysis')) 
    
    log_obj.set_id(id_process)
    log_obj.send(('imprime', 'identificador {} definido com sucesso'.format(id_process)))
    return redirect(url_for('main.new_analysis')) 

@main.route('/IMGprocessor', methods=['POST', 'GET'])
@login_required
def IMGprocessor():
    global file_searcher
    global log_obj
    
    if id_process is '': 
        flash('Defina o identificador da análise.'.format(id_process), 'error')
        return redirect(url_for('main.new_analysis')) 
    
    img_proc = ImageProcessor(file_searcher.files["images"])
    img_proc.process(True, 1, log_obj)
    return '', 204

@main.route('/VIDprocessor')
@login_required
def VIDprocessor():
    global file_searcher
    global log_obj

    if id_process is '': 
        flash('Defina o identificador da análise.'.format(id_process), 'error')
        return redirect(url_for('main.new_analysis')) 
    
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/IMGVIDprocessor')
@login_required
def IMGVIDprocessor():
    global file_searcher
    global log_obj
    
    if id_process is '': 
        flash('Defina o identificador da análise.'.format(id_process), 'error')
        return redirect(url_for('main.new_analysis')) 
    
    img_proc = ImageProcessor(file_searcher.files["images"])
    img_proc.process(True, 1, log_obj)
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

@main.route('/IMGreport', methods=['POST', 'GET'])
@login_required
def IMGreport():
    global log_obj
    
    img_report = Report(log_obj.log_path, log_obj.result_file) 
    html_path = img_report.generate_img(return_path=False)
    
    return render_template('report.html', html=[html_path], name=current_user.name, 
                           id_report=id_process, path=global_path) 
    
@main.route('/VIDreport', methods=['POST', 'GET'])
@login_required
def VIDreport():
    global log_obj
    
    vid_report = Report(log_obj.log_path, log_obj.result_file) 
    html_path = img_report.generate_vid_summary(return_path=False)
    
    return render_template('report.html', html=[html_path], name=current_user.name, 
                           id_report=id_process, path=global_path) 

@main.route('/IMGVIDreport', methods=['POST', 'GET'])
@login_required
def IMGVIDreport():
    global log_obj
    global id_process
    global global_path
    
    img_report = Report(log_obj.log_path, log_obj.result_file) 
    html_img = img_report.generate_img(return_path=False)
    
    vid_report = Report(log_obj.log_path, log_obj.result_file) 
    html_vid = img_report.generate_vid_summary(return_path=False)
    
    html_paths = [html_img, html_vid]
    return render_template('report.html', html=html_paths, name=current_user.name, 
                           id_report=id_process, path=global_path) 

@main.route('/search_analysis', methods=['POST', 'GET'])
@login_required
def search_analysis():
    return render_template('search_analysis.html')
