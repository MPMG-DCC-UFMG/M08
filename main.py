#!/usr/bin/env python
import sys, os, time, gc
from datetime import datetime
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sys.path.append('./M08')
from imageprocessor import ImageProcessor
from videoprocessor import VideoProcessor
from filesearcher import FileSearcher
from report import ReportImage, ReportVideo
from configcnn import ConfigCNN
from log import Log

from flask import Blueprint, render_template, request, jsonify,\
                    flash, redirect, url_for, render_template_string
from . import db
from . import dialog
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

#### Global variables
log_obj = Log()
file_searcher = None
id_process = ''
global_path = 'Caminho do Diretório'
conf = ConfigCNN.conf
img_report, vid_report = None, None
####

@main.route('/')
def index():
    return render_template('index.html')


####################################################
#                  BUSCAR ANALISE                  #
####################################################

@main.route('/search_analysis', methods=['POST', 'GET'])
@login_required
def search_analysis():
    global log_obj
    global id_process
    global conf
    
    return render_template('search_analysis.html', name=current_user.name, id_process=id_process, buffer=log_obj.buffer,  
                                                   conf_nsfw=conf['nsfw'], conf_face=conf['face'], 
                                                   conf_child=conf['child'], conf_age=conf['age'])

@main.route('/set_analysis')
@login_required
def set_analysis():
    global log_obj
    global id_process

    analysis_path = dialog._open_dialog_analysis()
    if analysis_path is not None and analysis_path != '':
        log_obj.send(('imprime', 
                      '{1} - [{0}] Importando dados de análise'.format(current_user.name,
                                                                     datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 
                     ))
        
        idx = analysis_path.rfind('.')
        id_process = os.path.basename(analysis_path[:idx])

        log_obj.set_id(id_process)
        return jsonify(id_process=id_process)
        
    flash('Erro ao carregar arquivo.'.format(id_process), 'error')
    return redirect(url_for('main.new_analysis')) 

####################################################
#                  NOVA ANALISE                    #
####################################################

@main.route('/new_analysis', methods=['POST', 'GET'])
@login_required
def new_analysis():
    global global_path
    global id_process
    global log_obj
    global conf
    
    if request.method == 'POST':
        print(request.form['info3'])
    return render_template('new_analysis.html', name=current_user.name, path=global_path, 
                                                id_process=id_process, buffer=log_obj.buffer,
                                                conf_nsfw=conf['nsfw'], conf_face=conf['face'], 
                                                conf_child=conf['child'], conf_age=conf['age'])

@main.route('/SOdialog')
@login_required
def SOdialog():
    global file_searcher
    global log_obj
    global global_path

    local_path = dialog._open_dialog_file()
    if local_path is not None and local_path != '':
        log_obj.send(('imprime', 
                      '{1} - [{9}] Nova análise'.format(current_user.name,
                                                     datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 
                     ))
        
        global_path = local_path
        file_searcher = FileSearcher(local_path)
        file_searcher.get_from_directory(log_obj, 0, verbose_fs=True) #signal_msg, task_id

        log_obj.set_rootpath(local_path)
    
    return jsonify(path=local_path)

@main.route('/idprocess', methods=['GET','POST'])
@login_required
def IDset():
    global id_process
    global log_obj

    id_ = request.form.get('id-process')
    
    if id_ in log_obj.all_logs: 
        flash('O identificador {} já existe.'.format(id_), 'error')
        return redirect(url_for('main.new_analysis')) 
    
    id_process = id_
    log_obj.set_id(id_process)
    return redirect(url_for('main.new_analysis')) 

@main.route('/log/<window>', methods=['POST', 'GET'])
@login_required
def refresh_log(window):
#     flash(log_obj.buffer, 'log')
    if window == 'search': return redirect(url_for('main.search_analysis'))
    else: return redirect(url_for('main.new_analysis')) 


@main.route('/IMGprocessor', methods=['POST', 'GET'])
@login_required
def IMGprocessor():
    global file_searcher
    global log_obj
    
    if id_process is '': 
        flash('Defina o identificador da análise.'.format(id_process), 'error')
        return redirect(url_for('main.new_analysis')) 
        
    img_proc = ImageProcessor(file_searcher.files["images"], log_obj)
    img_proc.process(batch_size=32)
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
    
    img_proc = ImageProcessor(file_searcher.files["images"], log_obj)
    img_proc.process(batch_size=32)
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)
    return '', 204

####################################################
#                      RELATÓRIO                   #
####################################################

@main.route('/settings/<window>', methods=['POST', 'GET'])
@login_required
def settings_search(window):
    
    global conf
    global log_obj
    
    conf['nsfw']  = float(request.form.get('conf_nsfw'))
    conf['face']  = float(request.form.get('conf_face'))
    conf['child'] = float(request.form.get('conf_child'))
    conf['age']   = float(request.form.get('conf_age'))
    
    log_obj.send(('imprime', 'Limiares definidos: '+str(conf) ))
    
    if window == 'search': return redirect(url_for('main.search_analysis'))
    else: return redirect(url_for('main.new_analysis'))  
    

@main.route('/IMGreport', methods=['POST', 'GET'])
@login_required
def IMGreport():
    global log_obj
    global id_process
    global img_report

    with open('./M08/templates/report_header.html', 'r', encoding='utf-8') as f:
        header = f.read()

    img_report = ReportImage(log_obj.log_path, id_process, log_obj,
                             conf_age=conf['age'], conf_child=conf['child'], 
                             conf_face=conf['face'], conf_nsfw=conf['nsfw']) 
    
    conteudo, id_tabela = img_report.generate_report(return_path=False)

    return render_template_string(header+conteudo+'{% endblock %}', id_report = id_process, 
                                  id_tabela=id_tabela, name=current_user.name, path=img_report.rootpath)

    
@main.route('/VIDreport', methods=['POST', 'GET'])
@login_required
def VIDreport():
    global log_obj
    global id_process
    global vid_report
    
    with open('./M08/templates/report_header.html', 'r', encoding='utf-8') as f:
        header = f.read()
    
    vid_report = ReportVideo(log_obj.log_path, id_process,log_obj,
                             conf_age=conf['age'], conf_child=conf['child'], 
                             conf_face=conf['face'], conf_nsfw=conf['nsfw'])  
    
    conteudo, id_tabela = vid_report.generate_report(return_path=False)

    return render_template_string(header+conteudo+'{% endblock %}', id_report = id_process, 
                                  id_tabela=id_tabela, name=current_user.name, path=vid_report.rootpath)

@main.route('/IMGVIDreport', methods=['POST', 'GET'])
@login_required
def IMGVIDreport():
    global log_obj
    global id_process
    global img_report
    global vid_report
    
    with open('./M08/templates/report_vid_img_header.html', 'r', encoding='utf-8') as f:
        header = f.read()
    
    img_report = ReportImage(log_obj.log_path, id_process, log_obj,
                             conf_age=conf['age'], conf_child=conf['child'], 
                             conf_face=conf['face'], conf_nsfw=conf['nsfw']) 
    
    html_img, id_tabela = img_report.generate_report(return_path=False)
    html_img = '<section class=\"section has-background-white\">' + \
                   '<h3 class=\"subtitle is-3 has-text-centered has-background-link-light pb-2\">Imagens</h3>' + html_img + \
                '</section>'
    
    vid_report = ReportVideo(log_obj.log_path, id_process, log_obj,
                             conf_age=conf['age'], conf_child=conf['child'], 
                             conf_face=conf['face'], conf_nsfw=conf['nsfw']) 
    
    html_vid, id_tabela_2 = vid_report.generate_report(return_path=False)
    html_vid = '<section class=\"section has-background-white\">' + \
                   '<h3 class=\"subtitle is-3 has-text-centered has-background-link-light pb-2\">Vídeos</h3>' + html_vid + \
                '</section>'
    
    
    return render_template_string(header+html_img+html_vid+'{% endblock %}', id_report = id_process, 
                                  id_tabela=id_tabela, id_tabela_2=id_tabela_2, name=current_user.name, 
                                  path=img_report.rootpath)
@main.route('/analysis_down', methods=['POST', 'GET'])
@login_required
def analysis_down():
    global img_report
    global vid_report
    global id_process
    global log_obj
    
    savepath = None
    
    savepath = dialog._open_dialog_file()
    savepath = os.path.join(savepath, id_process)
    if savepath is None or savepath == '':
        return '', 204
    
    log_obj.send(('imprime', 
                      '{1} - [{0}] '.format(current_user.name,datetime.now().strftime("%d/%m/%Y %H:%M:%S")) +
                      'Exportando dados.'
                 ))
    
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    
    if img_report is not None:
        img_report.html_style(savepath)
    if vid_report is not None:
        vid_report.html_style(savepath)
    
    log_obj.dump(savepath)
    img_report, vid_report = None, None
    gc.collect()
            
    return '', 204


@main.route('/showmedia/<path:img_url>', methods=['POST', 'GET']) 
def showmedia(img_url):
    
    img_url = str(Path(img_url)).encode('utf-8')
    dialog.show_local_image(img_url)
    return '', 204
    
