import sys, os, cv2
import datetime, time
import numpy as np
import random

import tensorflow as tf
from tensorflow.data import Dataset
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session

from filesearcher import FileSearcher
from log import Log
from faces import get_faces_mtcnn
from configcnn import ConfigCNN
from mtcnn_local.mtcnn import MTCNN
from tf_open_nsfw.model import OpenNsfwModel, InputType


class ImageProcessor():
    
    def __init__(self, files_dict, log):
        self.file_names = list(files_dict.keys())
        self.log = log
        
        # Models
        self.detector = detector = MTCNN() # intialized
        tf.reset_default_graph()
        self.model_nsfw = OpenNsfwModel()
        self.model_age  = model_from_json(open(ConfigCNN.model_architecture).read())

        self.batch_faces= {}
        self.conf_faces = {}
        
        self.VGG_MEAN = [104, 117, 123]
        self.nsfw_size = (256, 256)
    
    
    def process(self, batch_size=64, use_gpu=True):
        self.log.send(("imprime", 'Iniciando processamento de imagens. ' +  
                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
        
        # criando dataset para quebrar em batches
        dataset = Dataset.from_tensor_slices(self.file_names)
        dataset = dataset.map(lambda filename: tuple(tf.py_func(
                            self.load_img, [filename], (tf.uint8, filename.dtype)) ) )
        dataset = dataset.map(self.nsfw_preprocess).batch(batch_size)

        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        set_session(sess)

        iteration = 0
        with sess:
            
            # intialize models (expect MTCNN)
            self.model_nsfw.build(weights_path=ConfigCNN.nsfw_weights_path, 
                              input_type=InputType.TENSOR)
            sess.run(tf.global_variables_initializer())
            
            self.model_age.load_weights(ConfigCNN.model_weights)
            self.log.send(("imprime", 'Todos os modelos foram carregados. ' +  
                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
            
            iterator = dataset.make_one_shot_iterator()
            
            while True:
                try:
                    # Face detection runs here (self.load_img)
                    Xdata, filenames = sess.run(iterator.get_next())
                    
                    # NSFW predictions
                    prob_nsfw = sess.run(self.model_nsfw.predictions, feed_dict={self.model_nsfw.input: Xdata})
                    
                    # Age predictions
                    faces = np.concatenate(list(self.batch_faces.values()), axis=0)
                    predictions = self.model_age.predict(faces)
                    
                    # store batch predictions
                    num_faces = [len(v) for v in self.batch_faces.values()]
                    batch_idx = np.append([0], np.cumsum(num_faces))

                    count_imgs = 0
                    for k, filename in enumerate(filenames):
                        result = {'prob_nsfw': '', 'conf_faces': '', 'prob_age': '', 
                                  'prob_child':'', 'prob_gender': ''}
                        
                        filename = filename.decode("utf-8")
                        
                        result['prob_nsfw'] = prob_nsfw[k][1]
                        
                        if filename in self.batch_faces.keys():
                            idx = (batch_idx[count_imgs], num_faces[count_imgs])
                            result['conf_faces']  = self.conf_faces[filename] 
                            result['prob_age']    = predictions[0][idx[0]:idx[0]+idx[1]]
                            result['prob_child']  = predictions[1][idx[0]:idx[0]+idx[1]]
                            result['prob_gender'] = predictions[2][idx[0]:idx[0]+idx[1]]
                            count_imgs += 1
                    
                        self.log.send( ("data_file", filename, result) )
                    
                    iteration += len(filenames)
                    percentage = float(iteration)/len(self.file_names)
                    self.log.send(("imprime", 'Progresso {:.0f}%: '.format(percentage*100) + 
                                                 '|{:25}|'.format('#'*int(25*percentage)) +  
                                                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
                                    ))   
                except tf.errors.OutOfRangeError:
                    self.log.send(("imprime", 'Finalizado o processamento de imagens' + 
                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
                    
                    self.log.send(("finish",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
                    break
                except (KeyboardInterrupt, SystemExit):
                    self.log.send(("imprime", 'Processamento interrompido manualmente' +  
                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
                    
                    self.log.send(("finish", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
                    raise
            
    
    def load_img(self, filename):

        filename = filename.decode()
        default_return = np.zeros((100,100,3), dtype=np.int8)
        
        if not os.path.isfile(filename):
            self.log.send( ("imprime", "Arquivo não encontrado: {:}".format(filename)) )
            return default_return
        
        try:
            img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as ex:
            self.log.send(("imprime","Erro ao ler o arquivo {:}:{:}".format(filename, ex)))
            return default_return

        if img is None:
            self.log.send(("imprime","Erro ao ler o arquivo {:}".format(filename)))
            return default_return

        shape_img = img.shape
        # fix channels
        if len(shape_img) < 3: img = np.stack((img,)*3, axis=-1)
        elif shape_img[2] != 3: img = img[:,:,:3]
        shape_img = img.shape
        
        new_shape_img = shape_img
        min_dim, max_dim = min(shape_img[0], shape_img[1]), max(shape_img[0], shape_img[1])
        if min_dim < 30 or max_dim < 60:
            self.log.send(("imprime","Imagem é muito pequena {:}".format(filename)))
            return default_return

        lim_inferior, lim_superior = 720, 1440
        if not ( (max_dim > lim_inferior) and (max_dim < lim_superior) ):
            scale_factor = lim_inferior if (lim_inferior/max_dim) > 1 else lim_superior
            scale_factor /= max_dim
            new_shape_img = (int(shape_img[0] * scale_factor),
                             int(shape_img[1] * scale_factor) )
#             print('{:} rescaled by {:.2f}. Original size: {:}, New size: {:}'.format(filename,scale_factor, shape_img, new_shape_img))

        
        #### ARMAZENANDO FACES ####
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = get_faces_mtcnn(img, self.detector)
        
        if len(faces) > 0:
            self.conf_faces[filename]  = []
            self.batch_faces[filename] = []
            for tp in faces:
                fc, coord, confid_face = tp
                
                fc = cv2.resize(fc, (ConfigCNN.window_size[0], ConfigCNN.window_size[1]), interpolation=cv2.INTER_AREA)
                self.conf_faces[filename].append(confid_face)
                self.batch_faces[filename].append(fc)    
        
            self.batch_faces[filename] = np.array(self.batch_faces[filename]).astype('float32') / 255.
        
        img = cv2.resize(img, self.nsfw_size, interpolation=cv2.INTER_LINEAR)
        return img, filename
    
    def nsfw_preprocess(self, npy, filename):        
        
        npy.set_shape([None, None, None])
        image = tf.image.convert_image_dtype(npy, tf.float32, saturate=True)

        image = tf.image.resize_images(image, self.nsfw_size,
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True)
        
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
        
        image = tf.image.encode_jpeg(image, format='', quality=75,
                                     progressive=False, optimize_size=False,
                                     chroma_downsampling=True,
                                     density_unit=None,
                                     x_density=None, y_density=None,
                                     xmp_metadata=None)

        image = tf.image.decode_jpeg(image, channels=3,
                                     fancy_upscaling=False,
                                     dct_method="INTEGER_ACCURATE")

        image = tf.cast(image, dtype=tf.float32)

        image = tf.image.crop_to_bounding_box(image, 16, 16, 224, 224)

        image = tf.reverse(image, axis=[2])
        image -= self.VGG_MEAN
        return image, filename