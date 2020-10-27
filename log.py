import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, random
from datetime import datetime
from shutil import rmtree, copyfile
    

class Log():
    
    def __init__(self):
        
        log_path = os.path.join(os.getcwd(), 'M08', 'log')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.log_path = log_path
        
        all_logs = []
        for logs in os.listdir(log_path):
            all_logs.append(logs[:-4])
        self.all_logs = set(all_logs)
        
        self.id_analysis = None
        self.result_file = None
        
        today = datetime.now()
        self.logfile  = os.path.join(log_path, 'log_temp_{}.txt'.format( today.strftime("%Y%m%d_%H%M%S") ))
        
        self.results  = {'images': [], 'videos': [], 'rootpath': ''}
        self.buffer  = ''
    
    
    def set_rootpath(self, rootpath):
        self.results['rootpath'] = rootpath
    
    def set_id(self, id_analysis, empty=True):
        self.id_analysis = id_analysis
        self.result_file = os.path.join(self.log_path, id_analysis)
        self.result_file += '.npz'
        
        self.send(('imprime', 'Identificador da análise: {}'.format(id_analysis)))
        
    def send(self, tup):
        
        mode = tup[0]
        
        if mode == 'imprime':
            msg = tup[1] + '\n'
            self.buffer += msg 
            
            fp = open(self.logfile, 'a+')
            fp.write(msg)
            fp.close()
            
        elif mode == 'video_file':

            frames_video, target_file, timing_tmp = tup[1]
            
            result = {}
            result['Arquivo'] = target_file
            result['frames_video'] = frames_video
            result['Tempo de Análise'] = timing_tmp
            
            self.results['videos'].append(result)

    
        elif mode == 'data_file':
            
            result = {}
            result['Arquivo'] = tup[1]
            result['data'] = tup[2]
            result['Tempo de Análise'] = 0.0
               
            self.results['images'].append(result)
        
            
        elif mode == 'finish':                    
            if len(self.results['images']) > 0 or len(self.results['videos']) > 0:
                np.savez_compressed(self.result_file, images=self.results['images'] , 
                                                      videos=self.results['videos'],
                                                      rootpath=self.results['rootpath'])
                
    def dump(self, savepath):
        
        time_ = '_'.join(self.logfile.split('_')[-2:])
        time_ = time_[:-4]
        
        logpath = os.path.join(savepath, 'log_'+str(self.id_analysis)+'_'+time_+'.txt')
        copyfile(self.logfile, logpath)
        log.buffer = ''
        os.remove(self.logfile)
        
        npzpath = os.path.join(savepath, 'dados_'+str(self.id_analysis)+'_'+time_+'.npz')
        copyfile(self.result_file, npzpath)
        
        logdir = os.path.join(self.log_path, self.id_analysis)
        if os.path.isdir(logdir):
            rmtree(logdir)
            
            
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            