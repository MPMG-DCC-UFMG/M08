import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, random
    

class Log():
    
    def __init__(self,):
        
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
        
        self.logfile  = os.path.join(log_path, 'log_temp{}.txt'.format( int(random.random()*1e5) ))
        self.results  = {'images': [], 'videos': []}
        self.buffer  = ''
    
    def set_id(self, id_analysis):
        self.id_analysis = id_analysis
        self.result_file = os.path.join(self.log_path, id_analysis)
        
    def send(self, tup):
        
        mode = tup[0]
        
        if mode == 'imprime':
            msg = tup[1]
            self.buffer += msg + '\n'
            
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
            result['Tempo de Análise'] = tup[3]
               
            self.results['images'].append(result)
        
            
        elif mode == 'finish':                    
            #os.remove(self.logfile)
            np.savez_compressed(self.result_file, images=self.results['images'] , 
                                                  videos=self.results['videos'])
            self.result_file += '.npz'
            
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            