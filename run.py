import sys, os, time, gc
from shutil import copyfile, move
from datetime import datetime
from pathlib import Path
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
from report import ReportImage, ReportVideo
from configcnn import ConfigCNN
from log import Log

import argparse

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parser = argparse.ArgumentParser(description='M08: Detecção de Pedofilia Infantil em Imagens e Vídeos.')
parser.add_argument('-p', '--path', help='Diretório raíz onde se encontram as mídias para análise.', type=dir_path)
parser.add_argument('-i','--id', help='Identificador único (nome) da análise.')
parser.add_argument('-t', '--type', help='Tipo de dado a ser analisado [imagens | videos | todos]',
                    default='todos')
parser.add_argument('-o', '--output', help='Diretório para onde os resultados devem ser exportados.',
                    default='', type=dir_path)

args = parser.parse_args()
conf = ConfigCNN.conf
current_user = os.getenv('USERNAME')
current_user = current_user if current_user is not None else ''
log_obj = Log(id_analysis=args.id, rootpath=args.path, std_out=True)
img_report, vid_report = None, None

log_obj.send(('imprime',
                  '{1} - [{0}] '.format(current_user, datetime.now().strftime("%d/%m/%Y %H:%M:%S")) +
                  'Nova análise.'
             ))

file_searcher = FileSearcher(args.path)
file_searcher.get_from_directory(log_obj, 0, verbose_fs=True) #signal_msg, task_id

if args.type == 'imagens' or args.type == 'todos':
    img_proc = ImageProcessor(file_searcher.files["images"], log_obj)
    img_proc.process(batch_size=32)

    img_report = ReportImage(log_obj.log_path, log_obj.id_analysis, log_obj,
                             conf_age=conf['age'], conf_child=conf['child'],
                             conf_face=conf['face'], conf_nsfw=conf['nsfw'])
    img_html, id_ = img_report.generate_report(return_path=False)

if args.type == 'videos' or args.type == 'todos':
    vid_proc = VideoProcessor(file_searcher.files["videos"])
    vid_proc.process(log_obj)

    vid_report = ReportVideo(log_obj.log_path, log_obj.id_analysis, log_obj,
                             conf_age=conf['age'], conf_child=conf['child'],
                             conf_face=conf['face'], conf_nsfw=conf['nsfw'])
    vid_html, id_ = vid_report.generate_report(return_path=False)

if args.output != '':

    savepath = os.path.join(args.output, log_obj.id_analysis)
    log_obj.send(('imprime',
                      '{1} - [{0}] '.format(current_user, datetime.now().strftime("%d/%m/%Y %H:%M:%S")) +
                      'Exportando dados para {}'.format(savepath)
                 ))

    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    if img_report is not None:
        img_report.html_style(savepath)
    if vid_report is not None:
        vid_report.html_style(savepath)
    log_obj.dump(savepath)
