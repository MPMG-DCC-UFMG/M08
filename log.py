from configcnn import ConfigCNN
import matplotlib.pyplot as plt
import pandas as pd
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
        
        self.ageclasses = ConfigCNN.classes
        self.buffer = ''
        
    def send(self, tup):
        
        mode = tup[0]
        
        if mode == 'imprime':
            msg = tup[1]
            self.buffer += msg + '\n'
    
        elif mode == 'data_file':
            res = ['cont_age', 'cont_faixa', 'age_pred', 'child_pred', 
                   'prob_nsfw', 'idx_age_pred', 'idx_child_pred', 'all_preds',
                   'conf_faces']
            
            # Target File
            self.results['Arquivo'].append(tup[1]) 
            
            # Resultado
            
            for k, resultado in enumerate(tup[2]):
                if res[k] in self.attprint.keys():
                    
                    try:
                        if res[k] == 'idx_age_pred':
                            resultado = [self.ageclasses[age] for age in resultado]
                    except: resultado = str(resultado)
                    
                    try:
                        if res[k] == 'conf_faces':
                            resultado = ['{:.3f}'.format(conf) for conf in resultado]          
                    except: resultado = str(resultado)
                        
                    try:
                        if res[k] == 'prob_nsfw':
                            resultado = '{:.3f}'.format(resultado)        
                    except: resultado = str(resultado)
                    
                    self.results[self.attprint[res[k]]].append(resultado)
               
            # Elapsed time
            self.results['Tempo de Análise'].append('{:.2f}'.format(tup[3]))
            
        elif mode == 'finish':
#             self.buffer += 'Resultado da análise salvo em {}'.format(self.logfile[:-3]+'csv')                        
                                    
            fp = open(self.logfile, 'a+')
            fp.write(self.buffer)
            fp.close()
            
            results = pd.DataFrame.from_dict(self.results)
            results.to_csv(self.logfile[:-3]+'csv')
        
            