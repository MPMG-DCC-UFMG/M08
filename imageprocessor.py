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

        self.VGG_MEAN = [104, 117, 123]
        self.nsfw_size = (256, 256)
    
    def process(self, batch_size=64, use_gpu=True):
        self.log.send(("imprime", 'Iniciando processamento de imagens. ' +  
                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
        
        # criando dataset para quebrar em batches
        dataset = Dataset.from_tensor_slices(self.file_names)
        dataset = dataset.map(lambda filename: tuple(tf.py_func(
                            self.load_img, [filename], (tf.uint8, tf.float64, tf.float64, filename.dtype)) ))
        dataset = dataset.filter(lambda img, faces, conf, filename: tf.math.not_equal(filename,tf.constant('')))
        dataset = dataset.map(self.nsfw_preprocess)
        dataset = dataset.map(self.faces_ragged).map(self.conf_ragged).batch(batch_size)

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
                    start = time.time()
                    
                    ### Face detection runs here (self.load_img)
                    Xdata, Xfaces, conf_faces, filenames = sess.run(iterator.get_next())
                    
                    ### NSFW predictions
                    prob_nsfw = sess.run(self.model_nsfw.predictions, feed_dict={self.model_nsfw.input: Xdata})
                    
                    ### Age predictions                    
                    Xfaces = tf.squeeze(Xfaces, axis=1)
                    conf_faces = sess.run(tf.squeeze(conf_faces, axis=1).flat_values)
                    idx = sess.run(Xfaces.row_splits)

                    if -1 in conf_faces: 
                        mask = tf.math.not_equal( conf_faces, tf.constant(-1.) )
                        Xfaces = sess.run(tf.boolean_mask(Xfaces.flat_values, mask))
                        predictions = self.model_age.predict(Xfaces)
                    else:
                        predictions = self.model_age.predict(sess.run(Xfaces.flat_values))
                        
                    # store batch predictions
                    j = 0 # count predictions
                    for k, filename in enumerate(filenames):
                        result = {'prob_nsfw': '', 'conf_faces': '', 'prob_age': '', 
                                  'prob_child':'', 'prob_gender': ''}
                        
                        filename = filename.decode("utf-8")
                        
                        result['prob_nsfw'] = prob_nsfw[k][1]
                        
                        if conf_faces[idx[k]] != -1.:
                            num_faces = idx[k+1] - idx[k]
                            result['conf_faces']  = conf_faces[idx[k]:idx[k+1]] 
                            result['prob_age']    = predictions[0][j:j+num_faces]
                            result['prob_child']  = predictions[1][j:j+num_faces]
                            result['prob_gender'] = predictions[2][j:j+num_faces]
                            j += num_faces

                        self.log.send( ("data_file", filename, result) )
                    
                    end = time.time()
                    iteration += len(filenames)
                    percentage = float(iteration)/len(self.file_names)
                    self.log.send(("imprime", 'Progresso {:.0f}%: '.format(percentage*100)  + 
                                                 '|{:25}| '.format('#'*int(25*percentage))   +  
                                                 'Tempo decorrido:{:.3f}, '.format(end-start) +
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
        default_img  = np.zeros((1,self.nsfw_size[0],self.nsfw_size[1],3), dtype=np.uint8)
        default_face = np.zeros((1,ConfigCNN.window_size[0],ConfigCNN.window_size[0],3))
        default_conf = [-1.]
        
        if not os.path.isfile(filename):
            self.log.send( ("imprime", "Arquivo não encontrado: {:}".format(filename)) )
            return default_img, default_face, default_conf, ''
        
        try:
            img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as ex:
            self.log.send(("imprime","Erro ao ler o arquivo {:}: {:}".format(filename, ex)))
            return default_img, default_face, default_conf, ''

        if img is None:
            self.log.send(("imprime","Erro ao ler o arquivo {:}".format(filename)))
            return default_img, default_face, default_conf, ''

        # fix channels
        shape_img = img.shape
        if len(shape_img) < 3: img = np.stack((img,)*3, axis=-1)
        elif shape_img[2] != 3: img = img[:,:,:3]
        shape_img = img.shape
        
        new_shape_img = shape_img
        min_dim, max_dim = min(shape_img[0], shape_img[1]), max(shape_img[0], shape_img[1])
        if min_dim < 30 or max_dim < 60:
            self.log.send(("imprime","Imagem é muito pequena {:}".format(filename)))
            return default_img, default_face, default_conf, ''

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
        
        faces_list, conf_list = [], []
        if len(faces) > 0:
            for tp in faces:
                fc, coord, confid_face = tp
                conf_list.append(confid_face)
                
                fc = cv2.resize(fc, (ConfigCNN.window_size[0], ConfigCNN.window_size[1]), interpolation=cv2.INTER_AREA)
                faces_list.append(fc.astype('float64') / 255.)
        else: 
            faces_list = default_face
            conf_list  = default_conf
            
        img = cv2.resize(img, self.nsfw_size, interpolation=cv2.INTER_LINEAR)
        return img, faces_list, conf_list, filename
    
    
    def conf_ragged(self, image, faces, conf_npy, filename):
        conf = tf.ragged.stack(conf_npy)
        conf.set_shape([None, None, None])
        conf = tf.RaggedTensor.from_tensor(conf)
        conf = tf.cast(conf, dtype=tf.float32)
        return image, faces, conf, filename
        
    
    def faces_ragged(self, image, faces_npy, conf, filename):
        faces = tf.ragged.stack(faces_npy)
        faces.set_shape([None, None, None, None, None, None])  
        faces = tf.RaggedTensor.from_tensor(faces)
        faces = tf.cast(faces, dtype=tf.float32)

        return image, faces, conf, filename
        
    
    def nsfw_preprocess(self, npy, faces, conf, filename):        
        
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
        return image, faces, conf, filename