from configcnn import ConfigCNN
import pandas as pd
import numpy as np
import os, re


class Report():
    def __init__(self, rootpath, filename, conf_nsfw=0.3, max_age=13,):
        
        self.savepath = rootpath
        self.filename = filename[:-4]
        self.logfile  = np.load(os.path.join(rootpath, filename), allow_pickle=True)
        
        self.ageclasses = ConfigCNN.classes
        self.conf_nsfw = conf_nsfw
        self.max_age = [idx for idx, age in enumerate(self.ageclasses) if str(max_age) in age][0]
        
        self.res_order = ['cont_age', 'cont_faixa', 'age_pred', 'child_pred', 
                          'prob_nsfw', 'idx_age_pred', 'idx_child_pred', 'all_preds',
                          'conf_faces']
        
        self.results  = None
        self.classes  = ['Criança', 'NSFW']
        
    def generate_img(self, return_path=True):
        
        self.results  = {'Arquivo': [], 'Probabilidade NSFW': [], 'Número de Faces': [], 'Confiança Faces': [],  
                         'Faixas de Idade': [], 'Detector Criança': [], 'Classe': [], 'Tempo de Análise': []}
        
        results = self.logfile['images']
        
        for result in results:      
            classes = ''
            self.results['Arquivo'].append(result['Arquivo'])
            self.results['Tempo de Análise'].append(round(result['Tempo de Análise'], 2))

            nsfw = round(result['data'][self.res_order.index('prob_nsfw')], 3)
            self.results['Probabilidade NSFW'].append(nsfw)
            if nsfw >= self.conf_nsfw: classes += '-NSFW'
            
            self.results['Número de Faces'].append(result['data'][self.res_order.index('cont_age')])

            conf_face = result['data'][self.res_order.index('conf_faces')]
            self.results['Confiança Faces'].append([round(conf, 3) for conf in conf_face])

            idx_age = result['data'][self.res_order.index('idx_age_pred')]
            if idx_age is None: self.results['Faixas de Idade'].append(None)
            else: 
                self.results['Faixas de Idade'].append([self.ageclasses[age] for age in idx_age])
                if sum(idx_age <= self.max_age) > 0: classes += '-Criança' 
           
            idx_child = result['data'][self.res_order.index('idx_child_pred')]
            if idx_child is None: self.results['Detector Criança'].append(None)
            else: self.results['Detector Criança'].append([False if child == 1 else True for child in idx_child])
                
            self.results['Classe'].append(classes)

        report = self.html_style()
        
        if return_path:
            html_path = os.path.join(self.savepath, self.filename+'.html') 
            report.to_html(html_path)
            return html_path
        else:
            return report
    
    def generate_vid_summary(self, return_path=True):
        
        self.results  = {'Arquivo': [], 'Probabilidade NSFW': [], 'Número de Faces': [], 'Confiança Faces': [],  
                         'Faixas de Idade': [], 'Detector Criança': [], 'Classe': [], 'Tempo de Análise': []}
        
        results = self.logfile['videos']
        
        for result in results:
            classes = ''
            self.results['Arquivo'].append(result['Arquivo'])
            self.results['Tempo de Análise'].append(round(sum(result['Tempo de Análise']['all']), 2))
            
            prob_nsfw  = self.get_stats(result['frames_video'], self.res_order.index('prob_nsfw'))
            if prob_nsfw['Max'] >= self.conf_nsfw: classes += '-NSFW'
            self.results['Probabilidade NSFW'].append(prob_nsfw)
            
            num_faces = self.get_unique(result['frames_video'], self.res_order.index('cont_age'))
            self.results['Número de Faces'].append({'Média': round(np.mean(num_faces), 1), 
                                                    'Desvio': round(np.std(num_faces), 1)}  )

            conf_faces = self.get_stats(result['frames_video'], self.res_order.index('conf_faces'))
            self.results['Confiança Faces'].append(conf_faces)

            ages = self.get_unique(result['frames_video'], self.res_order.index('idx_age_pred'))
            self.results['Faixas de Idade'].append([self.ageclasses[age] for age in ages])
            if len(ages) > 0 and sum(ages <= self.max_age) > 0: classes += '-Criança' 
            
            conf_child = self.get_unique(result['frames_video'], self.res_order.index('idx_child_pred'))
            self.results['Detector Criança'].append([False if child == 1 else True for child in conf_child])
        
            self.results['Classe'].append(classes)
            
        report = self.html_style()
        
        if return_path:
            html_path = os.path.join(self.savepath, self.filename+'.html') 
            report.to_html(html_path)
            return html_path
        else:
            return report
        
    def generate_vid_perframe(self, filename):
        pass ########## TODO
    
    def get_unique(self, dic_frames, k):

        age_frames = [res[k] for frame, res in dic_frames.items() if isinstance(frame, int) and res[k] is not None ]
        if k == self.res_order.index('cont_age'): return np.unique(age_frames)
        
        all_ages = []
        for frame in age_frames:
            all_ages.extend([np.argmax(age) for age in frame])
            
        return np.unique(all_ages)
    
    def get_stats(self, dic_frames, k):
        
        frames = [res[k] for frame, res in dic_frames.items() if isinstance(frame, int) \
                                                              and res[k] is not None]
        if k == self.res_order.index('prob_nsfw'):
            return {'Min': round(np.min(frames), 2), 'Max': round(np.max(frames), 2),
                    'Média': round(np.mean(frames), 2), 'Desvio': round(np.std(frames), 2)}
        
        else:
            all_data = []
            for frame in frames:
                all_data.extend(data for data in frame)

            if len(all_data) == 0: return []
            return {'Min': round(np.min(all_data), 2), 'Max': round(np.max(all_data), 2),
                    'Média': round(np.mean(all_data), 2), 'Desvio': round(np.std(all_data), 2)}

        return []           
        
        
    def html_style(self,):
        styles = [
        dict(selector="th", props=[("font-size", "100%"),
                                   ("text-align", "center"),
                                   ("font-family", "Helvetica"),]),
        dict(selector="td", props=[("font-size", "100%"),
                                   ("text-align", "center"),
                                   ("font-family", "Helvetica")] ),
        ]
        
        log_df = pd.DataFrame(self.results)

        log_style = (log_df.style.apply(self.color_nsfw, axis=1)
                           .format({'Arquivo': self.make_clickable})
                           .set_table_styles(styles))

        html = log_style.render()
        return html
        

    def make_clickable(self, url):
        name = os.path.basename(url)
        return '<a href="{}">{}</a>'.format(url,name)

    def color_nsfw(self, data):

        colors = ['#bbdefb', '#ffecb3', '#ffcdd2']
        porn, child = False, False
        attr = ['' for i in range(len(data))]

        is_max = pd.Series(data=False, index=data.index)
        is_max['Classe'] = 'NSFW' in data['Classe'] 

        if is_max.any():
            porn = True
            attr = ['background-color: {:}'.format(colors[0]) for v in is_max]

        if data['Número de Faces'] > 0:
            is_max['Classe'] = 'Criança' in data['Classe'] 
            if is_max.any():
                color = 2 if porn else 1 
                attr = ['background-color: {:}'.format(colors[color]) for v in is_max]

        return attr
        