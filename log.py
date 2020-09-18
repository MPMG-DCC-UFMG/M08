from configcnn import ConfigCNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
    

class Log():
    
    def __init__(self, files_path, plot=False):
        self.files_path = files_path
        
        log_path = os.path.join(os.getcwd(), 'log')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        
        logs = os.listdir(log_path)
        
        self.logfile = os.path.join(log_path, 'log_{:03}.txt'.format(len(logs)//2))
        
        self.attprint = {'cont_age': 'Número de Faces', 'prob_nsfw': 'Probabilidade NSFW', 'idx_age_pred': 'Faixas de Idade',
                        'conf_faces': 'Confiança Faces'}
        
        self.results  = {'Arquivo': [], 'Probabilidade NSFW': [], 'Número de Faces': [], 'Confiança Faces': [],  
                         'Faixas de Idade': [], 'Tempo de Análise': []}
        
        self.res_order = ['cont_age', 'cont_faixa', 'age_pred', 'child_pred', 
                           'prob_nsfw', 'idx_age_pred', 'idx_child_pred', 'all_preds',
                           'conf_faces']
        
        self.ageclasses = ConfigCNN.classes
        self.buffer = ''
        
    def send(self, tup):
        
        mode = tup[0]
        
        if mode == 'imprime':
            msg = tup[1]
            self.buffer += msg + '\n'
            
        elif mode == 'video_file':

            frames_video, target_file, timing_tmp = tup[1]
            
            self.results['Arquivo'].append(target_file)
            self.results['Tempo de Análise'].append(round(sum(timing_tmp['all']), 2))
            
            ages = self.get_unique(frames_video, self.res_order.index('idx_age_pred'))
            self.results['Faixas de Idade'].append([self.ageclasses[age] for age in ages])

            conf_faces = self.get_hist(frames_video, self.res_order.index('conf_faces'))
#             self.results['Confiança Faces'].append([(round(edges), hist) for edges,hist in zip(conf_faces[1][1:], conf_faces[0])])
            self.results['Confiança Faces'].append(conf_faces)
            
            prob_nsfw  = self.get_hist(frames_video, self.res_order.index('prob_nsfw'))
#             self.results['Probabilidade NSFW'].append([(round(edges), hist) for edges,hist in zip(prob_nsfw[1][1:], prob_nsfw[0])])
            self.results['Probabilidade NSFW'].append(prob_nsfw)
                
            k = self.res_order.index('cont_age')
            num_faces  = []
            num_faces  = [ res[k]  for frame, res in frames_video.items() if isinstance(frame, int) and res[k] is not None]
            num_faces  = ( ('media', round(np.mean(num_faces),2)) , ('desvio', round(np.std(num_faces),2)) )
            self.results['Número de Faces'].append(num_faces)

    
        elif mode == 'data_file':
            
            # Target File
            self.results['Arquivo'].append(tup[1]) 
            
            # Resultado
            
            for k, resultado in enumerate(tup[2]):
                if self.res_order[k] in self.attprint.keys():
                    
                    try:
                        if self.res_order[k] == 'idx_age_pred':
                            resultado = [self.ageclasses[age] for age in resultado]
                    except: resultado = str(resultado)
                    
                    try:
                        if self.res_order[k] == 'conf_faces':
                            resultado = ['{:.3f}'.format(conf) for conf in resultado]          
                    except: resultado = str(resultado)
                        
                    try:
                        if self.res_order[k] == 'prob_nsfw':
                            resultado = '{:.3f}'.format(resultado)        
                    except: resultado = str(resultado)
                    
                    self.results[self.attprint[self.res_order[k]]].append(resultado)
               
            # Elapsed time
            self.results['Tempo de Análise'].append('{:.2f}'.format(tup[3]))
            
        elif mode == 'finish':
#             self.buffer += 'Resultado da análise salvo em {}'.format(self.logfile[:-3]+'csv')                        
            
            fp = open(self.logfile, 'a+')
            fp.write(self.buffer)
            fp.close()
            self.buffer = ''
            
            results = pd.DataFrame.from_dict(self.results)
            results.to_csv(self.logfile[:-3]+'csv')
        
    def get_unique(self, dic_frames, k):

        age_frames = [res[k] for frame, res in dic_frames.items() if isinstance(frame, int) and res[k] is not None ]
        
        all_ages = []
        for frame in age_frames:
            all_ages.extend([np.argmax(age) for age in frame])
            
        return np.unique(all_ages)
    
    def get_hist(self, dic_frames, k):
        
        frames = [res[k] for frame, res in dic_frames.items() if isinstance(frame, int) \
                                                              and res[k] is not None]
        if k == self.res_order.index('prob_nsfw'):
#             hist = np.histogram(frames, bins=9, range=(0,1), density=False)
            return ( ('min', np.min(frames)), ('max', np.max(frames)), ('media', np.mean(frames)), 
                     ('desvio', np.std(frames)) ) 
            
        else:
            all_data = []
            for frame in frames:
                all_data.extend(data for data in frame)
#             hist = np.histogram(all_data, bins=10, range=(0,1), density=False)
            if len(all_data) == 0: return []
            return ( ('min', np.min(all_data)), ('max', np.max(all_data)), ('media', np.mean(all_data)), 
                 ('desvio', np.std(all_data)) ) 

        return []           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            