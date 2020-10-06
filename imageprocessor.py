import sys
import datetime
import time
import numpy as np
import os.path
import cv2
from faces import get_faces_mtcnn
from configcnn import ConfigCNN
import random


class ImageProcessor():

    def __init__(self, files_dict):
        self.file_names = list(files_dict.keys())
        self.timing = {"detect_faces": [], "get_faces_mtcnn2": [], "nsfw": [], "age": [], "all": []}

    def conv_pred(self, p, verbose=False):
        if p.shape[-1] > 1:
            p = p.argmax(axis=-1)
        else:
            if verbose:
                print("convpred p.shape <= 1")
            p = (proba > 0.5).astype('int32')
        return p

    def get_data_and_predictions2(self, file_name, model_age, sess, model_nsfw, fn_load_image,
                                  detector, timing, child_conn, verbose=False):

        if not os.path.isfile(file_name):
            print("not os.path.isfile({:})".format(file_name))
            return 0, 0, None, None, -1., None, None, [], []
        cont = 0
        img_work = None
        try:
            img_work = cv2.imread(file_name)
        except Exception as ex:
            if verbose:
                child_conn.send(("imprime", "Erro em cv2.imread: {:}".format(img_name)))
                print(ex)
            return 0, 0, None, None, -1., None, None, [], []

        if img_work is None:
            if verbose: child_conn.send(("imprime", "img_work=None em cv2.imread: {:}".format(img_name)))
            return 0, 0, None, None, -1., None, None, [], []

        shape_img = img_work.shape

        min_dim = min(shape_img[0], shape_img[1])
        if min_dim < 30:
            print("min_dim < 30")
            return 0, 0, None, None, 0., None, None, [], []

        max_dim = max(shape_img[0], shape_img[1])
        if max_dim < 60:
            print("max_dim < 60")
            return 0, 0, None, None, 0., None, None, [], []

        lim_inferior = max(720, max_dim)
        if lim_inferior > max_dim:
            new_shape_img = (int((shape_img[1] * lim_inferior) / max_dim),
                             int((shape_img[0] * lim_inferior) / max_dim))
            img_work = cv2.resize(img_work, new_shape_img, interpolation=cv2.INTER_AREA)
            #print("upscaled")
        else:
            lim_superior = min(1440, max_dim)
            if max_dim > 1440 and lim_superior < max_dim:
                new_shape_img = (int((shape_img[1] * lim_superior) / max_dim),
                                 int((shape_img[0] * lim_superior) / max_dim))
                img_work = cv2.resize(img_work, new_shape_img, interpolation=cv2.INTER_AREA)
                #print("downscaled")
            else:
                new_shape_img = shape_img
                #print("not scaled")

        img_work = cv2.cvtColor(img_work, cv2.COLOR_BGR2RGB)
        area_total = img_work.shape[0] * img_work.shape[1]

        t1 = datetime.datetime.now()
        faces = []
#         try:
            #print(shape_img, img_work.shape, img_name)
        faces = get_faces_mtcnn(img_work, detector, timing["detect_faces"])
#         except Exception as e:
#             print("Erro em mtcnn", e)
#             print(base_path, img_name)
#             print(img_work.shape)
#             raise Exception(e)

        t2 = datetime.datetime.now()
        timing["get_faces_mtcnn2"].append((t2 - t1).total_seconds())

        tf1 = datetime.datetime.now()
        adj_faces = []
        coordinates = []
        conf_faces = []
        for tp in faces:
            fc, coord, confid_face = tp
            area_face = (coord[2] - coord[0]) * (coord[3] - coord[1])
            prop_x = (coord[3] - coord[1]) / new_shape_img[1]
            if area_face < 1200 and prop_x < 0.06:
                continue
            coordinates.append(coord)
            conf_faces.append(confid_face)
            cont += 1

            adj_face = cv2.resize(fc, (ConfigCNN.window_size[0] + 4, ConfigCNN.window_size[1] + 4), interpolation=cv2.INTER_AREA)
            adj_faces.append(adj_face[2:-2, 2:-2])

        tf2 = t1 = datetime.datetime.now()

        prob_nsfw = -1.0
        try:
            image = fn_load_image(file_name)
            predictions = sess.run(model_nsfw.predictions, feed_dict={model_nsfw.input: image})
            prob_nsfw = predictions[0][1]
        except Exception as ex:
            if verbose:
                child_conn.send(("imprime", "erro model_nsfw.predictions em: {:}".format(img_name)))
                print(ex)

        t2 = datetime.datetime.now()
        timing["nsfw"].append((t2 - t1).total_seconds())

        age_pred = None
        child_pred = None
        idx_age_pred = None
        idx_child_pred = None
        all_preds = []
        if cont > 0:
            # predict
            t1 = datetime.datetime.now()
            all_preds = model_age.predict(np.array(adj_faces).astype('float32') / 255.)
            t2 = datetime.datetime.now()
            timing["age"].append((tf2 - tf1 + t2 - t1).total_seconds())
            age_pred = all_preds[0]
            child_pred = all_preds[1]
            idx_age_pred = self.conv_pred(all_preds[0])
            idx_child_pred = self.conv_pred(all_preds[1])
            cont_age = len(idx_age_pred[idx_age_pred < len(ConfigCNN.classes)])
            cont_faixa = len(idx_child_pred[idx_child_pred < len(ConfigCNN.faixa_child_adult)])
        else:
            cont_age = cont_faixa = 0

        return (cont_age, cont_faixa, age_pred, child_pred, prob_nsfw, idx_age_pred, idx_child_pred, all_preds,conf_faces)

    def process(self, use_gpu, total_processes, child_conn, verbose=False):

        from keras import backend as K
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        from keras.models import model_from_json

        config = tf.ConfigProto()
        child_conn.send(("imprime", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))

        if use_gpu:
            fracao = {1: 0.770, 2: 0.380, 3: 0.26, 4: 0.19}
            config.gpu_options.per_process_gpu_memory_fraction = fracao[total_processes]
        else:
            config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                    inter_op_parallelism_threads=1, allow_soft_placement=True)

        sess = tf.Session(config=config)
        set_session(sess)

        from mtcnn.mtcnn import MTCNN
        from tf_open_nsfw.model import OpenNsfwModel, InputType
        from tf_open_nsfw.image_utils import create_tensorflow_image_loader
        from tf_open_nsfw.image_utils import create_yahoo_image_loader

        detector = MTCNN()

        tf.reset_default_graph()
        model_nsfw = OpenNsfwModel()
        image_loader = ConfigCNN.IMAGE_LOADER_YAHOO
        input_type = InputType.TENSOR
        weights_path = ConfigCNN.nsfw_weights_path

        with sess:
            model_nsfw.build(weights_path=weights_path, input_type=input_type)
            fn_load_image = None
            if input_type == InputType['TENSOR']:
                if image_loader == ConfigCNN.IMAGE_LOADER_TENSORFLOW:
                    fn_load_image = create_tensorflow_image_loader(sess)
                else:
                    fn_load_image = create_yahoo_image_loader()
            sess.run(tf.global_variables_initializer())

            model_age = model_from_json(open(ConfigCNN.model_architecture).read())
            model_age.load_weights(ConfigCNN.model_weights)
            t0 = datetime.datetime.now()
            child_conn.send(("imprime", "Modelos carregados. Inicia processamento de imagens."))

            abort = False
            del_mtcnn = 0
            child_conn.send(('imprime','Iniciando análise de imagens - {}'.format(
                                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")) ))
            for k, target_file in enumerate(sorted(self.file_names)):
                print('\rProcessando imagem {0}/{1}'.format(k, len(self.file_names)), end='', flush=True )
                
                if target_file[-3:] == 'txt': continue

                if not os.path.isfile(target_file): continue
                t3 = datetime.datetime.now()
                res = self.get_data_and_predictions2(target_file, model_age, sess, model_nsfw, 
                                                     fn_load_image,detector, self.timing, child_conn, verbose)
                time.sleep(random.randint(1,10)/1000)
                t4 = datetime.datetime.now()
                tempo = (t4 - t3).total_seconds()

                child_conn.send(("data_file", target_file, res, tempo))
                del_mtcnn += 1


            child_conn.send(("imprime", '{} - {}'.format("Finalizado",
                                                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z"))))
            try:
                child_conn.send(("finish", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
            except Exception as e:
                print("erro finish")
            child_conn.send(('imprime', 'Análise de imagens completa. Verifique o log em caso de erros.'))


        try:
            #print("del models")
            del detector
            del model_nsfw
        except Exception as e:
            print("ex del models")

        try:
            #print("K.clear_session()")
            K.clear_session()
        except Exception as e:
            print("excecao K.clear_session()")
        time.sleep(1)
