from configcnn import ConfigCNN
import pandas as pd
import numpy as np
import os, re

class Report:
    
    def __init__(self, rootpath, filename, conf_age=0.8, conf_child=0.6, conf_face=0.8, conf_nsfw=0.3):
        self.savepath = rootpath
        self.filename = filename[:-4]
        self.logfile  = np.load(os.path.join(rootpath, filename), allow_pickle=True)
        
        self.conf = {'age': conf_age, 'child': conf_child, 'face': conf_face, 'nsfw': conf_nsfw}
        self.results  = {}

    def apply_confidence(self, data):

        num_faces, idades, num_criancas = 0, '', 0

        
        if len(data['conf_faces']) > 0:
            
            mask_faces = np.array(data['conf_faces']) > self.conf['face']
            num_faces = np.sum(mask_faces)
            
            prob_age = [prob for k, prob in enumerate(data['prob_age']) if mask_faces[k] == True]
            prob_age = [prob for prob in data['prob_age'] if max(prob) > self.conf['age']]
            if len(prob_age) > 0: 
                idades   = [ConfigCNN.classes[age] for age in np.argmax(prob_age, axis=-1)]
            
            prob_child = [prob for k, prob in enumerate(data['prob_child']) if mask_faces[k] == True] 
            prob_child = [prob for prob in data['prob_child'] if max(prob) > self.conf['child']]
            if len(prob_child) > 0:
                num_criancas = np.sum( [False if child == 1 else True for child in np.argmax(prob_child, axis=-1)] ) 
            
        return num_faces, idades, num_criancas    
    
    def generate_img(self, return_path=True):
        
        self.results  = {'Arquivo': [], 'NSFW': [], 'Faces': [],  
                         'Idades': [], 'Crianças': [], 'Classe': []}
        
        results = self.logfile['images']
        
        for result in results:      
            classes = ''
            
            self.results['Arquivo'].append(result['Arquivo'])
            
            # NSFW
            nsfw = np.round(result['data']['prob_nsfw'], 3)
            self.results['NSFW'].append(nsfw)
            if nsfw >= self.conf['nsfw']: classes += 'Pode conter pornografia. '
            
            # Número de Faces, Idades, Número de Crianças
            num_faces, idades, num_criancas = self.apply_confidence(result['data']) 
            self.results['Faces'].append(num_faces)
            self.results['Idades'].append(idades)
            self.results['Crianças'].append(num_criancas)
            
            if num_criancas > 0: classes += 'Pode conter menores de idade.'
            
            
            self.results['Classe'].append(classes)
            
        report, table_id = self.html_style()
        
        if return_path:
            html_path = os.path.join(self.savepath, self.filename+'.html') 
            report.to_html(html_path)
            return html_path, table_id
        else:
            return report, table_id
        
        
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

        html = log_style.render(table_id=self.filename)

        idx = html.find('table id="')
        table_id = html[idx:].split('\"')[1]
        return html, '#' + table_id
    
    def make_clickable(self, url):
        name = os.path.basename(url)
        return '<a href="{}">{}</a>'.format(url,name)

    def color_nsfw(self, data):

        colors = ['#bbdefb', '#ffecb3', '#ffcdd2']
        porn, child = False, False
        attr = ['' for i in range(len(data))]

        is_max = pd.Series(data=False, index=data.index)
        is_max['Classe'] = 'pornografia' in data['Classe'] 

        if is_max.any():
            porn = True
            attr = ['background-color: {:}'.format(colors[0]) for v in is_max]

        is_max['Classe'] = 'menores' in data['Classe'] 
        if is_max.any():
            color = 2 if porn else 1 
            attr = ['background-color: {:}'.format(colors[color]) for v in is_max]

        return attr

