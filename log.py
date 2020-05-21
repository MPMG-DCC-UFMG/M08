from configcnn import ConfigCNN
import matplotlib.pyplot as plt
import os

class Log():
    
    def __init__(self, files_path, plot=False):
        self.files_path = files_path
        self.logfile = os.path.join(files_path, 'log.txt')
        
        self.attprint = {'cont_age': '# Pessoas', 'prob_nsfw': 'Probabilidade NSFW', 'idx_age_pred': 'Faixas de Idade:'}
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
            self.buffer += 10*'-' + tup[1] + 10*'-' + '\n'
            
            # Resultado
            for k, resultado in enumerate(tup[2]):
                if res[k] in self.attprint.keys():
                    if res[k] == 'idx_age_pred':
                        resultado = [self.ageclasses[age] for age in resultado]
                    self.buffer += self.attprint[res[k]] + ': ' + str(resultado) + '\n'
               
            self.buffer += 'Elapsed time: ' + str(tup[3]) + '\n'
            
        elif mode == 'finish':
            fp = open(self.logfile, 'w')
            fp.write(self.buffer)
            fp.close()
            
        
            