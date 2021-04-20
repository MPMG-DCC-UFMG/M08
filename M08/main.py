#!/usr/bin/env python
import sys, os, re, time, gc
import humanize, stat, json, mimetypes
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
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

from flask import Flask, Blueprint, render_template, request, jsonify,\
                  flash, redirect, url_for, render_template_string,\
                  current_app, make_response, session, send_file, Response
from flask.views import MethodView
from . import db
# from . import dialog
from flask_login import login_required, current_user


main = Blueprint('main', __name__)

####################################################
#                 VARIAVEIS GLOBAIS                #
####################################################
root, key = '', ''
def set_root(search=False):

    if search:
        root = os.path.join(os.getcwd(), 'M08', 'log')
    else:
        root = str(Path.home().parent)
    key = os.getenv('FS_KEY')
    return root, key

log_obj = Log()
kind = 'new' # [new | search | down]
file_searcher = None
id_process = ''
global_path = 'Caminho do Diretório'
conf = ConfigCNN.conf
img_report, vid_report = None, None
##########################################################

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
    global root
    global key
    global kind

    root, key = set_root(search=True)
    kind = 'search'

    return render_template('search_analysis.html', name=current_user.name, id_process=id_process, buffer=log_obj.buffer,
                                                   conf_nsfw=conf['nsfw'], conf_face=conf['face'],
                                                   conf_child=conf['child'], conf_age=conf['age'])

@main.route('/set_analysis')
@login_required
def set_analysis():
    global log_obj
    global id_process
    global conf

    path_name = request.args.get('path_name')
    entry_name = request.args.get('entry_name')

    analysis_path = path_name+entry_name

    if os.path.isdir(analysis_path):
        files = os.listdir(analysis_path)
        analysis_name = [name for name in files if '.npz' in name]

        if len(analysis_name) == 0:
            flash('{} não contém uma análise válida'.format(analysis_path), 'error')
            return redirect(url_for('main.path_view', p=path_name))

        log_obj.send(('imprime',
                      '{1} - [{0}] Importando dados de análise'.format(current_user.name,
                                                                     datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                     ))

        analysis_path = os.path.join(analysis_path, analysis_name[0])
        idx = analysis_path.rfind('.')
        id_process = os.path.basename(analysis_path[:idx])
        log_obj.set_id(id_process, search=True)

        return render_template('search_analysis.html', name=current_user.name, id_process=id_process, buffer=log_obj.buffer,
                                                       conf_nsfw=conf['nsfw'], conf_face=conf['face'],
                                                       conf_child=conf['child'], conf_age=conf['age'])

    else:
        flash('{} não é um diretório.'.format(analysis_path), 'error')
        return redirect(url_for('main.path_view', p=path_name))


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
    global root
    global key
    global kind

    if request.method == 'POST':
        print(request.form['info3'])

    root, key = set_root()
    kind = 'new'
    return render_template('new_analysis.html', name=current_user.name, path=global_path,
                                                id_process=id_process, buffer=log_obj.buffer,
                                                conf_nsfw=conf['nsfw'], conf_face=conf['face'],
                                                conf_child=conf['child'], conf_age=conf['age'])

# BROWSER UI #
@main.route('/set_new_analysis', methods=['POST', 'GET'])
@login_required
def set_new_analysis():
    global global_path
    global file_searcher
    global id_process
    global log_obj
    global conf

    path_name = request.args.get('path_name')
    entry_name = request.args.get('entry_name')
    global_path = path_name+entry_name

    if os.path.isdir(global_path):
        log_obj.send(('imprime',
                      '{1} - [{0}] Nova análise'.format(current_user.name,
                                                     datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                     ))

        file_searcher = FileSearcher(global_path)
        file_searcher.get_from_directory(log_obj, 0, verbose_fs=True) #signal_msg, task_id

        log_obj.set_rootpath(global_path)

        # return render_template('new_analysis.html', name=current_user.name)
        return render_template('new_analysis.html', name=current_user.name, path=global_path,
                                                    id_process=id_process, buffer=log_obj.buffer,
                                                    conf_nsfw=conf['nsfw'], conf_face=conf['face'],
                                                    conf_child=conf['child'], conf_age=conf['age'])

    else:
        flash('{} não é um diretório.'.format(global_path), 'error')
        return redirect(url_for('main.path_view', p=global_path))


# # OS UI #
# @main.route('/SOdialog')
# @login_required
# def SOdialog():
#     global file_searcher
#     global log_obj
#     global global_path

#     local_path = dialog._open_dialog_file()
#     if local_path is not None and local_path != '' and isinstance(local_path, str):
#         log_obj.send(('imprime',
#                       '{1} - [{0}] Nova análise'.format(current_user.name,
#                                                      datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#                      ))

#         global_path = local_path
#         file_searcher = FileSearcher(local_path)
#         file_searcher.get_from_directory(log_obj, 0, verbose_fs=True) #signal_msg, task_id

#         log_obj.set_rootpath(local_path)

#     return jsonify(path=local_path)

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
    global id_process

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
    global id_process

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
    html_img =   '<h3 class=\"subtitle is-3 has-text-centered has-background-white pb-2\">Imagens</h3>' + html_img #+ \

    vid_report = ReportVideo(log_obj.log_path, id_process, log_obj,
                             conf_age=conf['age'], conf_child=conf['child'],
                             conf_face=conf['face'], conf_nsfw=conf['nsfw'])

    html_vid, id_tabela_2 = vid_report.generate_report(return_path=False)
    html_vid = '<h3 class=\"subtitle is-3 mt-3 has-text-centered has-background-white pb-2\">Vídeos</h3>' + html_vid #+ \


    return render_template_string(header+html_img+html_vid+'{% endblock %}', id_report = id_process,
                                  id_tabela=id_tabela, id_tabela_2=id_tabela_2, name=current_user.name,
                                  path=img_report.rootpath)

@main.route('/analysis_down', methods=['POST', 'GET'])
@login_required
def analysis_down():
    global kind
    global root
    global key
    
    kind = 'down'
    root, key = set_root()
    return redirect(url_for('main.path_view'))
    


@main.route('/save_analysis', methods=['POST', 'GET'])
@login_required
def save_analysis():
    global img_report
    global vid_report
    global id_process
    global log_obj

    path_name = request.args.get('path_name')
    entry_name = request.args.get('entry_name')
    savepath = path_name+entry_name

    savepath = os.path.join(savepath, id_process)

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

    img_report, vid_report = None, None
    gc.collect()

    log_obj.dump(savepath)

    flash('Dados da análise salvos em {}'.format(savepath), 'success')
    return redirect(url_for('main.path_view', p=path_name))

@main.route('/showmedia/<path:img_url>', methods=['POST', 'GET'])
def showmedia(img_url):

    if 'linux' in sys.platform:
        img_url = str(Path('/' + unquote(img_url)))
    else:
        img_url = str(Path(unquote(img_url)))

#     dialog.show_local_image(img_url)
    res = send_file(img_url, as_attachment=False)
    return res


####################################################
#                SISTEMA DE ARQUIVOS               #
####################################################
if sys.platform == 'win32':
    root = os.path.normpath(os.getenv('HOMEPATH', ''))
else:
    root = os.path.normpath(os.getenv('FS_PATH', ''))
key = os.getenv('FS_KEY')

ignored = ['.bzr', '$RECYCLE.BIN', '.DAV', '.DS_Store', '.git', '.hg', '.htaccess', '.htpasswd', '.Spotlight-V100', '.svn', '__MACOSX', 'ehthumbs.db', 'robots.txt', 'Thumbs.db', 'thumbs.tps']
datatypes = {'audio': 'm4a,mp3,oga,ogg,webma,wav', 'archive': '7z,zip,rar,gz,tar', 'image': 'gif,ico,jpe,jpeg,jpg,png,svg,webp', 'pdf': 'pdf', 'quicktime': '3g2,3gp,3gp2,3gpp,mov,qt', 'source': 'atom,bat,bash,c,cmd,coffee,css,hml,js,json,java,less,markdown,md,php,pl,py,rb,rss,sass,scpt,swift,scss,sh,xml,yml,plist', 'text': 'txt', 'video': 'mp4,m4v,ogv,webm', 'website': 'htm,html,mhtm,mhtml,xhtm,xhtml'}
icontypes = {'fa-music': 'm4a,mp3,oga,ogg,webma,wav', 'fa-archive': '7z,zip,rar,gz,tar', 'fa-picture-o': 'gif,ico,jpe,jpeg,jpg,png,svg,webp', 'fa-file-text': 'pdf', 'fa-film': '3g2,3gp,3gp2,3gpp,mov,qt', 'fa-code': 'atom,plist,bat,bash,c,cmd,coffee,css,hml,js,json,java,less,markdown,md,php,pl,py,rb,rss,sass,scpt,swift,scss,sh,xml,yml', 'fa-file-text-o': 'txt', 'fa-film': 'mp4,m4v,ogv,webm', 'fa-globe': 'htm,html,mhtm,mhtml,xhtm,xhtml'}

@main.app_template_filter('size_fmt')
def size_fmt(size):
    return humanize.naturalsize(size)

@main.app_template_filter('time_fmt')
def time_desc(timestamp):
    mdate = datetime.fromtimestamp(timestamp)
    str = mdate.strftime('%Y-%m-%d %H:%M:%S')
    return str

@main.app_template_filter('data_fmt')
def data_fmt(filename):
    t = 'unknown'
    for type, exts in datatypes.items():
        if filename.split('.')[-1] in exts:
            t = type
    return t

@main.app_template_filter('icon_fmt')
def icon_fmt(filename):
    i = 'fa-file-o'
    for icon, exts in icontypes.items():
        if filename.split('.')[-1] in exts:
            i = icon
    return i

@main.app_template_filter('humanize')
def time_humanize(timestamp):
    mdate = datetime.utcfromtimestamp(timestamp)
    return humanize.naturaltime(mdate)

def get_type(mode):
    if stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        type = 'dir'
    else:
        type = 'file'
    return type

def partial_response(path, start, end=None):
    file_size = os.path.getsize(path)

    if end is None:
        end = file_size - start - 1
    end = min(end, file_size - 1)
    length = end - start + 1

    with open(path, 'rb') as fd:
        fd.seek(start)
        bytes = fd.read(length)
    assert len(bytes) == length

    response = Response(
        bytes,
        206,
        mimetype=mimetypes.guess_type(path)[0],
        direct_passthrough=True,
    )
    response.headers.add(
        'Content-Range', 'bytes {0}-{1}/{2}'.format(
            start, end, file_size,
        ),
    )
    response.headers.add(
        'Accept-Ranges', 'bytes'
    )
    return response

def get_range(request):
    range = request.headers.get('Range')
    m = re.match('bytes=(?P<start>\d+)-(?P<end>\d+)?', range)
    if m:
        start = m.group('start')
        end = m.group('end')
        start = int(start)
        if end is not None:
            end = int(end)
        return start, end
    else:
        return 0, None

class PathView(MethodView):
    # kind = new analysis (new) or search previous analysis (search)
    def get(self, p=''):
        
        global kind
        hide_dotfile = request.args.get('hide-dotfile', request.cookies.get('hide-dotfile', 'no'))

        path = os.path.join(root, p)
        if os.path.isdir(path):
            contents = []
            
            total = {'size': 0, 'dir': 0, 'file': 0}
            for filename in os.listdir(path):
                if filename in ignored:
                    continue
                if hide_dotfile == 'yes' and filename[0] == '.':
                    continue
                filepath = os.path.join(path, filename)
                stat_res = os.stat(filepath)
                info = {}
                info['name'] = filename
                info['mtime'] = stat_res.st_mtime
                ft = get_type(stat_res.st_mode)
                info['type'] = ft
                total[ft] += 1
                sz = stat_res.st_size
                info['size'] = sz
                total['size'] += sz
                contents.append(info)
            page = render_template('busca_arquivo.html', path=p, contents=contents, total=total,
                                                        hide_dotfile=hide_dotfile,real_path=path,
                                                        kind=kind)
            res = make_response(page, 200)
            res.set_cookie('hide-dotfile', hide_dotfile, max_age=16070400)
        elif os.path.isfile(path):
            if 'Range' in request.headers:
                start, end = get_range(request)
                res = partial_response(path, start, end)
            else:
                res = send_file(path)
                res.headers.add('Content-Disposition', 'attachment')
        else:
            res = make_response('Not found', 404)

        return res

    def put(self, p=''):
        if request.cookies.get('auth_cookie') == key:
            path = os.path.join(root, p)
            dir_path = os.path.dirname(path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            info = {}
            if os.path.isdir(dir_path):
                try:
                    filename = secure_filename(os.path.basename(path))
                    with open(os.path.join(dir_path, filename), 'wb') as f:
                        f.write(request.stream.read())
                except Exception as e:
                    info['status'] = 'error'
                    info['msg'] = str(e)
                else:
                    info['status'] = 'success'
                    info['msg'] = 'File Saved'
            else:
                info['status'] = 'error'
                info['msg'] = 'Invalid Operation'
            res = make_response(json.JSONEncoder().encode(info), 201)
            res.headers.add('Content-type', 'application/json')
        else:
            info = {}
            info['status'] = 'error'
            info['msg'] = 'Authentication failed'
            res = make_response(json.JSONEncoder().encode(info), 401)
            res.headers.add('Content-type', 'application/json')
        return res

    def post(self, p=''):
        if request.cookies.get('auth_cookie') == key:
            path = os.path.join(root, p)
            Path(path).mkdir(parents=True, exist_ok=True)

            info = {}
            if os.path.isdir(path):
                files = request.files.getlist('files[]')
                for file in files:
                    try:
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(path, filename))
                    except Exception as e:
                        info['status'] = 'error'
                        info['msg'] = str(e)
                    else:
                        info['status'] = 'success'
                        info['msg'] = 'File Saved'
            else:
                info['status'] = 'error'
                info['msg'] = 'Invalid Operation'
            res = make_response(json.JSONEncoder().encode(info), 200)
            res.headers.add('Content-type', 'application/json')
        else:
            info = {}
            info['status'] = 'error'
            info['msg'] = 'Authentication failed'
            res = make_response(json.JSONEncoder().encode(info), 401)
            res.headers.add('Content-type', 'application/json')
        return res

    def delete(self, p=''):
        if request.cookies.get('auth_cookie') == key:
            path = os.path.join(root, p)
            dir_path = os.path.dirname(path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            info = {}
            if os.path.isdir(dir_path):
                try:
                    filename = secure_filename(os.path.basename(path))
                    os.remove(os.path.join(dir_path, filename))
                    os.rmdir(dir_path)
                except Exception as e:
                    info['status'] = 'error'
                    info['msg'] = str(e)
                else:
                    info['status'] = 'success'
                    info['msg'] = 'File Deleted'
            else:
                info['status'] = 'error'
                info['msg'] = 'Invalid Operation'
            res = make_response(json.JSONEncoder().encode(info), 204)
            res.headers.add('Content-type', 'application/json')
        else:
            info = {}
            info['status'] = 'error'
            info['msg'] = 'Authentication failed'
            res = make_response(json.JSONEncoder().encode(info), 401)
            res.headers.add('Content-type', 'application/json')
        return res

path_view = PathView.as_view('path_view')
main.add_url_rule('/arquivos', view_func=path_view)
main.add_url_rule('/<path:p>', view_func=path_view)
