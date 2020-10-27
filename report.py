from configcnn import ConfigCNN
import pandas as pd
import numpy as np
import os, re, cv2
from pathlib import Path
from datetime import datetime
from time import gmtime, strftime

class ReportImage():
    
    def __init__(self, logpath, filename, log_obj, conf_age=0.7, conf_child=0.7, conf_face=0.8, conf_nsfw=0.3):
        self.savepath = logpath
        self.filename = filename
        
        self.log_obj = log_obj
                     
        self.logfile = None
        if os.path.isfile(os.path.join(logpath, filename+'.npz')):
            self.logfile  = np.load(os.path.join(logpath, filename+'.npz'), allow_pickle=True)
                   
        self.rootpath = self.logfile['rootpath']
        self.conf = {'age': conf_age, 'child': conf_child, 'face': conf_face, 'nsfw': conf_nsfw}
        self.results  = {}

    def apply_confidence(self, data):

        num_faces, idades, num_criancas = 0, [], 0

        
        if len(data['conf_faces']) > 0:

            mask_faces = np.array(data['conf_faces']) > self.conf['face']
            num_faces = np.sum(mask_faces)
            
            prob_age = data['prob_age'][mask_faces]
            prob_age = prob_age[np.max(prob_age, axis=-1) > self.conf['age']]
            if len(prob_age) > 0: 
                idades   = [ConfigCNN.classes[age] for age in sorted(np.argmax(prob_age, axis=-1)) ]
            idades += ['ND']* (num_faces-len(prob_age))
            
            prob_child = data['prob_child'][mask_faces]
            prob_child = prob_child[np.max(prob_child, axis=-1) > self.conf['child']]
            if len(prob_child) > 0:
                num_criancas = np.sum( [False if child == 1 else True for child in np.argmax(prob_child, axis=-1)] ) 
        
        return num_faces, idades, num_criancas    
    
    def generate_report(self, return_path=True):
        
        self.results  = {'Arquivo': [], 'NSFW': [], 'Faces': [],  
                         'Idades': [], 'Crianças': [], 'Classe': []}
        
        self.log_obj.send(('imprime', 
                           '{} - Iniciando criação de relatório de imagens'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                          ))
        
        if self.logfile is not None: 
            results = self.logfile['images']

            for result in results:      
                classes = ''

                self.results['Arquivo'].append(result['Arquivo'])

                # NSFW
                nsfw = np.round(result['data']['prob_nsfw'], 3)
                self.results['NSFW'].append('{:.3f}'.format(nsfw))
                if nsfw >= self.conf['nsfw']: classes += 'Pode conter pornografia. '

                # Número de Faces, Idades, Número de Crianças
                num_faces, idades, num_criancas = self.apply_confidence(result['data']) 
                self.results['Faces'].append(num_faces)
                self.results['Idades'].append(idades)
                self.results['Crianças'].append(num_criancas)

                if num_criancas > 0: classes += 'Pode conter menores de idade.'


                self.results['Classe'].append(classes)

        self.log_obj.send(('imprime', 
                           '{} - Relatório concluído. Criando página.'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 
                          ))
        
        report, table_id = self.html_style()             
        if return_path:
            html_path = os.path.join(self.savepath, self.filename+'.html') 
            report.to_html(html_path)
            return html_path, table_id
        else:
            return report, table_id
        
        
    def html_style(self, excel_path=None):
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

        if excel_path is not None:
            time_ = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_style.to_excel(os.path.join(excel_path, self.filename+'_'+time_+'_imagens.xlsx') )
            return

        html = log_style.render(table_id=self.filename)
        idx = html.find('table id="')
        table_id = html[idx:].split('\"')[1]
        return html, '#' + table_id
    
    def make_clickable(self, url):
        name = os.path.basename(url)
#         return '<a href="{}">{}</a>'.format(url,name)
        url = url.replace('\\', '\\\\')
        return '<a href=\"{{{{ url_for(\'main.showmedia\' , img_url=\'{}\') }}}}\"> {} </a>'.format(url, name)

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
        
        
    
class ReportVideo():
    
    def __init__(self, logpath, filename, log_obj, conf_age=0.8, conf_child=0.6, conf_face=0.8, conf_nsfw=0.3):
        self.savepath = logpath
        self.filename = filename

        self.log_obj = log_obj
        
        self.logfile = None
        if os.path.isfile(os.path.join(logpath, filename+'.npz')):
            self.logfile  = np.load(os.path.join(logpath, filename+'.npz'), allow_pickle=True)
        
        self.rootpath = self.logfile['rootpath']
        self.conf = {'age': conf_age, 'child': conf_child, 'face': conf_face, 'nsfw': conf_nsfw}
        self.max_frames = 8
        self.results  = {}

    def apply_confidence(self, data):

        num_faces, idades, num_criancas, probs = 0, '', 0, 0.
        
        if len(data['conf_faces']) > 0:
            
            mask_faces = np.array(data['conf_faces']) > self.conf['face']
            num_faces = np.sum(mask_faces)
            
            prob_age = data['prob_age'][mask_faces]
            prob_age = prob_age[np.max(prob_age, axis=-1) > self.conf['age']]
            if len(prob_age) > 0: 
                idades   = [age for age in np.argmax(prob_age, axis=-1)]
            
            prob_child = data['prob_child'][mask_faces]
            prob_child = prob_child[np.max(prob_child, axis=-1) > self.conf['child']]
            if len(prob_child) > 0:
                num_criancas = np.sum( [False if child == 1 else True for child in np.argmax(prob_child, axis=-1)] ) 
                probs = np.max( [prob[0] for prob in prob_child] )
            
        return data['prob_nsfw'], idades, num_criancas, probs   
    
    def generate_report(self, return_path=True):
        
        self.results  = {'Arquivo': [], 'Timestamp': [], 'Thumbnail': [], 'Classe': []}
        
        self.log_obj.send(('imprime', 
                           '{} - Iniciando criação de relatório de vídeos'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                          ))              
        if self.logfile is not None: 
            videos = self.logfile['videos']

            for video in videos: 

                results = {'frame':[], 'Porn':[], 'Ages':[], 'Num children':[], 
                            'Prob child': []} 

                frames = video['frames_video']
                for frame in frames.keys():
                    if not isinstance(frame, int): continue

                    nsfw, ages, num_child, prob_child  = self.apply_confidence(frames[frame])
                    results['frame'].append(frame)
                    results['Porn'].append(nsfw)
                    results['Ages'].append(ages)
                    results['Num children'].append(num_child)
                    results['Prob child'].append(prob_child)

                
                results = pd.DataFrame(results)
                porn  = results[results['Porn'] > self.conf['nsfw']] 
                porn  = porn[porn['Num children'] == 0]
                porn  = porn.sort_values('Porn', ascending=False) 

                child = results[results['Num children'] > 0]         
                child_porn = child[child['Porn'] > self.conf['nsfw']]
                child_porn = child_porn.sort_values('Porn', ascending=False) 

                retimages = []
                for row in child_porn.index: 
                    if len(retimages) >= self.max_frames: break
                    idx = child_porn.loc[row]['frame']
                    self.results['Arquivo'].append(video['Arquivo'])
                    retimages.append(idx)   
                    self.results['Classe'].append('Pode conter pornografia. Pode conter menores de idade.')

                for row in porn.index:
                    if len(retimages) >= self.max_frames: break
                    idx = porn.loc[row]['frame']
                    self.results['Arquivo'].append(video['Arquivo'])
                    retimages.append(idx)   
                    self.results['Classe'].append('Pode conter pornografia.')


                child = child.sort_values('Prob child', ascending=False)
                for row in child.index:
                    if len(retimages) >= self.max_frames: break
                    idx = child.loc[row]['frame']
                    self.results['Arquivo'].append(video['Arquivo'])
                    retimages.append(idx)   
                    self.results['Classe'].append('Pode conter menores de idade.')

                results.sort_values('Porn', ascending=False)
                for row in results.index:
                    if len(retimages) >= self.max_frames: break
                    idx = results.loc[row]['frame']
                    self.results['Arquivo'].append(video['Arquivo'])
                    retimages.append(idx)  
                    self.results['Classe'].append([''])
    
                self.results['Thumbnail'].extend(self.get_labeled_frames(video['Arquivo'], 
                                                                         retimages, frames))
                
                retimages = [ strftime("%H:%M:%S", gmtime(fr/float(video['frames_video']["fps"]))) for fr in retimages] 
                self.results['Timestamp'].extend(retimages)
        
        self.log_obj.send(('imprime', 
                           '{} - Relatório concluído. Criando página.'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 
                          ))
         
        report, table_id = self.html_style()
        
        if return_path:
            html_path = os.path.join(self.savepath, self.filename+'.html') 
            report.to_html(html_path)
            return html_path, table_id
        else:
            return report, table_id
        
    
    def get_labeled_frames(self, filename, frames, all_data):
        
        cap = cv2.VideoCapture(filename)
        retframes = []
        
        for pos_frame in frames:
            data = all_data[pos_frame]
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
            ret, frame = cap.read()
            if not ret:
                retframes.append('')
                continue
            
            shp = frame.shape
            font = cv2.FONT_HERSHEY_COMPLEX
            maximo = max(shp[0], shp[1])
            lim = min(1800, maximo)
            novoshp = (int((shp[1] * lim) / maximo), int((shp[0] * lim) / maximo))
            img2 = cv2.resize(frame, novoshp, interpolation=cv2.INTER_AREA) ###
            imgwork = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            areatotal = imgwork.shape[0] * imgwork.shape[1]
            
            cv2.putText(imgwork, "Porn: {:2.1f}%".format(100 * data['prob_nsfw']), (15, int(img2.shape[0] / 10)),
                            font, img2.shape[1] / 600, (255, 80, 50), 2, cv2.LINE_AA)
            
            if len(data['conf_faces']) > 0:
                for i in range(0, len(data['coords'])):
                    if data['conf_faces'][i] < self.conf['face']: continue
                    
                    (x, y, x2, y2) = data['coords'][i]
                
                    gender = np.argmax(data['prob_gender'][i], axis=-1)
                    if ConfigCNN.genero[gender] == 'M':
                        cor = (255, 90, 90)
                    else:
                        cor = (180, 180, 255)
                        
                    cv2.rectangle(imgwork, (x, y), (x2, y2), cor, 2)
                    x2proporcional = (x2 - x) * (novoshp[1] / shp[1])
                    szfont = min(200, max((x2 - x), 55)) / 150
                    if x2proporcional < 50: szfont = szfont * max(1, min((x2 - x) / x2proporcional, 1.35))
                    bold = 1

                    cor = None
                    label_child_adult = "ND"
                    
                    if np.max(data['prob_child'][i], axis=-1) > self.conf['child']: 
                        label_child_adult = ConfigCNN.faixa[np.argmax(data['prob_child'][i], axis=-1)] 
                    
                    if label_child_adult == 'Cr':
                        cor = (50, 50, 255)
                    elif label_child_adult == 'Ad':
                        cor = (50, 255, 120)
                    else:
                        cor = (210, 210, 210)
                        
                    cv2.putText(imgwork, label_child_adult, (max(x, 10), max(y, 20)), + font, szfont, cor,
                                bold, cv2.LINE_AA)
                    
                save_dir = os.path.join(self.savepath, self.filename)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                save_file = os.path.join(save_dir, os.path.basename(filename) + '_' + str(pos_frame) + '.jpg') 

                cv2.imwrite(save_file, imgwork)
                retframes.append(save_file)
                    
        return retframes
        
        
    def html_style(self,excel_path=None):
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
#                            .format({'Thumbnail': self.path_to_image_html})
                           .format({'Thumbnail': self.make_clickable})
                           .format({'Arquivo': self.make_clickable})
                           .set_table_styles(styles))

        if excel_path is not None:
            time_ = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_style.to_excel(os.path.join(excel_path, self.filename+'_'+time_+'_videos.xlsx') )
            return
        
        html = log_style.render(table_id=self.filename)

        idx = html.find('table id="')
        table_id = html[idx:].split('\"')[1]
        return html, '#' + table_id
    
    def make_clickable(self, url):
        name = os.path.basename(url)
        url = url.replace('\\', '\\\\')
        return '<a href=\"{{{{ url_for(\'main.showmedia\' , img_url=\'{}\') }}}}\"> {} </a>'.format(url, name)

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