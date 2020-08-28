import sys
import datetime
import time
import numpy as np
import os.path
import cv2
from PIL import Image
from io import BytesIO
import skimage
import skimage.io
import subprocess
from faces import get_faces_mtcnn
import random
from configcnn import ConfigCNN


class VideoProcessor:

    def __init__(self, source_path, files_path, dir_model=""):
        self.source_path = source_path
        self.files_path = files_path
        self.timing = {"detect_faces": [], "get_faces_mtcnn2": [], "nsfw": [], "age": [], "all": []}

    def conv_pred(p, verbose=False):
        if p.shape[-1] > 1:
            p = p.argmax(axis=-1)
        else:
            if verbose:
                print("self.conv_pred p.shape <= 1")
            p = (proba > 0.5).astype('int32')
        return p

    @staticmethod
    def get_data_and_predictions_frame(frame, frame_number, model_age, sess, model_nsfw, fn_load_image, detector,
                                       timing, show_img=False, return_img=False, parameter_confidence=None):
        cont = 0
        img = frame.copy()
        font = cv2.FONT_HERSHEY_COMPLEX
        if img is None:
            print("img=None frame", frame_number)
            return None

        shp = img.shape
        maximo = max(shp[0], shp[1])
        lim = min(1800, maximo)
        novoshp = (int((shp[1] * lim) / maximo), int((shp[0] * lim) / maximo))
        img2 = cv2.resize(img, novoshp, interpolation=cv2.INTER_AREA)
        imgwork = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        areatotal = imgwork.shape[0] * imgwork.shape[1]
        t1 = datetime.datetime.now()
        prob_nsfw = -1.0
        try:
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if im.mode != "RGB":
                im = im.convert('RGB')
            imr = im.resize((256, 256), resample=Image.BILINEAR)
            fh_im = BytesIO()
            imr.save(fh_im, format='JPEG')
            fh_im.seek(0)
            image = (skimage.img_as_float(skimage.io.imread(fh_im, as_grey=False)).astype(np.float32))
            H, W, _ = image.shape
            h, w = (224, 224)
            h_off = max((H - h) // 2, 0)
            w_off = max((W - w) // 2, 0)
            image = image[h_off:h_off + h, w_off:w_off + w, :]
            # RGB to BGR
            image = image[:, :, :: -1]
            image = image.astype(np.float32, copy=False)
            image = image * 255.0
            VGG_MEAN = [104, 117, 123]
            image -= np.array(VGG_MEAN, dtype=np.float32)
            image = np.expand_dims(image, axis=0)
            predictions = sess.run(model_nsfw.predictions, feed_dict={model_nsfw.input: image})
            prob_nsfw = predictions[0][1]
        except Exception as ex:
            print("erro model_nsfw.predictions em frame", frame_number)
            print(ex)

        if prob_nsfw < 0.1:
            if show_img or return_img:
                # cv2.putText( img2, str(int(100*prob_nsfw)), (20, 30), font, 1, (255, 255, 100), 2, cv2.LINE_AA)
                cv2.putText(img2, "Porn: {:2.1f}%".format(100 * prob_nsfw), (15, int(img2.shape[0] / 10)),
                            font, img2.shape[1] / 600, (255, 80, 50), 2, cv2.LINE_AA)
                if show_img:
                    cv2.namedWindow("video")
                    cv2.moveWindow("video", 0, 0)
                    # cv2.resizeWindow(str(frame_number), novoshp[0], novoshp[1])
                    # img2 = cv2.resize(img2, novoshp, interpolation=cv2.INTER_AREA)
                    cv2.imshow("video", img2)
                    cv2.waitKey(1000)
                    # cv2.destroyAllWindows()
                if return_img:
                    return 0, None, 0, 0, None, None, prob_nsfw, None, None, None, None, img2
                else:
                    return 0, None, 0, 0, None, None, prob_nsfw, None, None, None, None, None

        t2 = datetime.datetime.now()
        timing["nsfw"].append((t2 - t1).total_seconds())
        t1 = datetime.datetime.now()
        faces = get_faces_mtcnn(imgwork, detector, timing["detect_faces"])
        t2 = datetime.datetime.now()
        timing["get_faces_mtcnn2"].append((t2 - t1).total_seconds())
        tf1 = datetime.datetime.now()
        adjfaces = []
        coords = []
        conf_faces = []
        
        for tp in faces:
            fc, coord, confid_face = tp
            if parameter_confidence is not None:
                if confid_face < parameter_confidence[2]:
                    continue
            area_face = (coord[2] - coord[0]) * (coord[3] - coord[1])
            prop_x = (coord[3] - coord[1]) / novoshp[1]
            if area_face < 1200 and prop_x < 0.06:
                # print("area pequena", id_arq, area_face, prop_x)
                continue
            coords.append(coord)
            conf_faces.append(confid_face)
            cont += 1
            adjfc = cv2.resize(fc, (ConfigCNN.window_size[0] + 4, ConfigCNN.window_size[1] + 4), interpolation=cv2.INTER_AREA)
            adjfaces.append(adjfc[2:-2, 2:-2])
            
        tf2 = datetime.datetime.now()
        age_pred = None
        child_pred = None
        idx_age_pred = None
        idx_child_pred = None
        allpreds = []

        if (cont > 0):
            # predict
            t1 = datetime.datetime.now()
            allpreds = model_age.predict(np.array(adjfaces).astype('float32') / 255.)
            t2 = datetime.datetime.now()
            timing["age"].append((tf2 - tf1 + t2 - t1).total_seconds())
            age_pred = allpreds[0]
            child_pred = allpreds[1]
            idx_age_pred = VideoProcessor.conv_pred(allpreds[0])
            idx_child_pred = VideoProcessor.conv_pred(allpreds[1])
            cont_age = len(idx_age_pred[idx_age_pred < len(ConfigCNN.classes)])
            cont_faixa = len(idx_child_pred[idx_child_pred < len(ConfigCNN.faixa_child_adult)])

            if show_img or return_img:
                pred_age = VideoProcessor.conv_pred(allpreds[0])
                pred_child = VideoProcessor.conv_pred(allpreds[1])
                pred_gender = VideoProcessor.conv_pred(allpreds[2])
                # a linha a seguir é para impor confidencia > 'conf_child' na faixa de idade
                child_pred_max = allpreds[1][np.arange(idx_child_pred.shape[0]), idx_child_pred]
                # idx_child_pred_tmp = np.copy(idx_child_pred)
                # idx_child_pred_tmp[child_pred_max<parameter_confidence[1]] = len(faixa_child_adult)+1
                shp = novoshp
                maximo = max(shp[0], shp[1])
                lim = min(800, maximo)
                novoshp = (int((shp[0] * lim) / maximo), int((shp[1] * lim) / maximo))
                for i in range(0, len(coords)):
                    # descarta faces com child_pred abaixo do valor de confiança
                    if parameter_confidence is not None:
                        if child_pred_max[i] < parameter_confidence[1]:
                            continue
                    (x, y, x2, y2) = coords[i]
                    # print("area", (x2-x)*(y2-y), (100*(x2-x)*(y2-y))/areatotal)
                    class_pred = int(pred_age[i])
                    if ConfigCNN.genero[pred_gender[i]] == 'M':
                        cor = (255, 90, 90)
                    else:
                        cor = (180, 180, 255)

                    cv2.rectangle(img2, (x, y), (x2, y2), cor, 2)
                    # print(genero[pred_gender[i]], faixa[pred_child[i]], classes[class_pred])
                    x2proporcional = (x2 - x) * (novoshp[1] / shp[1])
                    # print((x2-x), x2proporcional, novoshp[1], shp[1])
                    szfont = min(200, max((x2 - x), 55)) / 150
                    if x2proporcional < 50: szfont = szfont * max(1, min((x2 - x) / x2proporcional, 1.35))
                    # if x2proporcional < 35: szfont = szfont * 1.1
                    bold = 1
                    if (szfont > 0.5): bold = 2
                    if ConfigCNN.faixa[pred_child[i]] == 'Cr':
                        cor = (50, 50, 255)
                    else:
                        cor = (50, 255, 120)
                    # cv2.putText( img2, faixa[pred_child[i]] + "-" + genero[pred_gender[i]] + " " +\
                    #            classes[class_pred], (max(x, 10), max(y, 20)), font, szfont, cor, bold, cv2.LINE_AA)
                    cv2.putText(img2, ConfigCNN.faixa[pred_child[i]], (max(x, 10), max(y, 20)), + font, szfont, cor,
                                bold, cv2.LINE_AA)
        else:
            cont_age = cont_faixa = 0

        ret_img = None
        if show_img or return_img:
            # cv2.putText( img2, str(int(100*prob_nsfw)), (20, 30), font, 1, (255, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(img2, "Porn: {:2.1f}%".format(100 * prob_nsfw), (15, int(img2.shape[0] / 10)),
                        font, img2.shape[1] / 600, (255, 80, 50), 2, cv2.LINE_AA)
            if show_img:
                cv2.namedWindow("video")
                cv2.moveWindow("video", 0, 0)
                # cv2.resizeWindow(str(frame_number), novoshp[0], novoshp[1])
                # img2 = cv2.resize(img2, novoshp, interpolation=cv2.INTER_AREA)
                cv2.imshow("video", img2)
                cv2.waitKey(1000)
                # cv2.destroyAllWindows()
            if return_img:
                ret_img = img2

        return cont, coords, cont_age, cont_faixa, age_pred, child_pred, prob_nsfw, idx_age_pred, idx_child_pred, \
               allpreds, conf_faces, ret_img

    @staticmethod
    def get_labeled_frame(res, frame, frame_number, show_img=False, return_img=False, parameter_confidence=None):
        cont = 0
        img = frame.copy()
        font = cv2.FONT_HERSHEY_COMPLEX

        if img is None:
            print("img=None frame", frame_number)
            return None

        shp = img.shape
        maximo = max(shp[0], shp[1])
        lim = min(1800, maximo)
        novoshp = (int((shp[1] * lim) / maximo), int((shp[0] * lim) / maximo))
        img2 = cv2.resize(img, novoshp, interpolation=cv2.INTER_AREA)
        imgwork = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        areatotal = imgwork.shape[0] * imgwork.shape[1]

        cont, coords, cont_age, cont_faixa, age_pred, child_pred, prob_nsfw, idx_age_pred, idx_child_pred, \
        allpreds, conf_faces, ret_img = res

        if prob_nsfw < 0.1:
            if show_img or return_img:
                # cv2.putText( img2, str(int(100*prob_nsfw)), (20, 30), font, 1, (255, 255, 100), 2, cv2.LINE_AA)
                cv2.putText(img2, "Porn: {:2.1f}%".format(100 * prob_nsfw), (15, int(img2.shape[0] / 10)),
                            font, img2.shape[1] / 600, (255, 80, 50), 2, cv2.LINE_AA)
                if show_img:
                    cv2.namedWindow("video")
                    cv2.moveWindow("video", 0, 0)
                    cv2.imshow("video", img2)
                    cv2.waitKey(1000)
                if return_img:
                    return img2
                else:
                    return None

        if (cont > 0):
            if show_img or return_img:
                pred_age = VideoProcessor.conv_pred(allpreds[0])
                pred_child = VideoProcessor.conv_pred(allpreds[1])
                pred_gender = VideoProcessor.conv_pred(allpreds[2])
                # a linha a seguir é para impor confidencia > 'conf_child' na faixa de idade
                child_pred_max = allpreds[1][np.arange(idx_child_pred.shape[0]), idx_child_pred]
                shp = novoshp
                maximo = max(shp[0], shp[1])
                lim = min(800, maximo)
                novoshp = (int((shp[0] * lim) / maximo), int((shp[1] * lim) / maximo))
                for i in range(0, len(coords)):
                    # não faz marcação em faces com conf_face abaixo do valor de confiança
                    if parameter_confidence is not None:
                        if conf_faces[i] < parameter_confidence[2]:
                            continue

                    (x, y, x2, y2) = coords[i]

                    class_pred = int(pred_age[i])
                    if ConfigCNN.genero[pred_gender[i]] == 'M':
                        cor = (255, 90, 90)
                    else:
                        cor = (180, 180, 255)

                    cv2.rectangle(img2, (x, y), (x2, y2), cor, 2)
                    x2proporcional = (x2 - x) * (novoshp[1] / shp[1])
                    szfont = min(200, max((x2 - x), 55)) / 150
                    if x2proporcional < 50: szfont = szfont * max(1, min((x2 - x) / x2proporcional, 1.35))
                    bold = 1

                    cor = None
                    label_child_adult = "ND"

                    # marca como ND faces com child_pred abaixo do valor de confiança
                    if parameter_confidence is not None:
                        if child_pred_max[i] < parameter_confidence[1]:
                            cor = (210, 210, 210)

                    if (szfont > 0.5):
                        bold = 2

                    if cor is None:
                        label_child_adult = ConfigCNN.faixa[pred_child[i]]
                        if ConfigCNN.faixa[pred_child[i]] == 'Cr':
                            cor = (50, 50, 255)
                        else:
                            cor = (50, 255, 120)

                    cv2.putText(img2, label_child_adult, (max(x, 10), max(y, 20)), + font, szfont, cor,
                                bold, cv2.LINE_AA)

        ret_img = None
        if show_img or return_img:
            cv2.putText(img2, "Porn: {:2.1f}%".format(100 * prob_nsfw), (15, int(img2.shape[0] / 10)),
                        font, img2.shape[1] / 600, (255, 80, 50), 2, cv2.LINE_AA)
            if show_img:
                cv2.namedWindow("video")
                cv2.moveWindow("video", 0, 0)
                cv2.imshow("video", img2)
                cv2.waitKey(1000)
            if return_img:
                ret_img = img2

        return ret_img

    def get_samples(cap):
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames < 0:
            return None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return None, None, None

        duration = float(num_frames) / float(fps)

        if duration < 20.:
            num_samples = int(max(duration, num_frames / 10))
            if num_samples < 2: num_samples = 2
            if num_samples > 20: num_samples = 20
        elif duration < 120.:
            num_samples = 20 + int((duration - 20) / 5)
        elif duration < 720.:
            num_samples = 40 + int((duration - 120) / 10)
        else:
            num_samples = 100 + int((duration - 720) / 30)

        if num_samples > num_frames:
            num_samples = num_frames

        min_range = 5
        if num_frames < 80:
            min_range = min(3, int((num_samples - 6) / 2))
            min_range = max(0, min_range)
        elif duration > 10:
            min_range = int(fps / 2)
        elif duration > 5:
            min_range = int(fps / 3)

        smp = random.sample(range(min_range, num_frames - min_range), num_samples)

        #print("(min_range, num_frames - min_range, int((num_frames - 2 * min_range) / (num_samples - 1)))")
        #division = - 1
        #try:
        #    division = int((num_frames - 2 * min_range) / (num_samples - 1))
        #except Exception as e:
        #    pass
        # print("min_rng={:}, num_frms{:}, num_smpls={:}, div={:}".format(min_range, num_frames, num_samples, division))

        if num_samples == 1:
            num_samples = 2
            print("num_samples == 1")
        try:
            for i in range(min_range, num_frames - min_range, int((num_frames - 2 * min_range) / (num_samples - 1))):
                smp.append(i)
        except Exception as e:
            print("Erro add samples", e)

        smp = sorted(smp)
        return smp, num_frames, fps

    def convert_video(self, video_input, video_output):
        try:
            os.remove(video_output)
        except:
            pass
        try:
            cmds = ['ffmpeg', '-i', video_input, video_output]
            print(cmds)
            proc = subprocess.Popen(cmds)
            proc.wait()
            print("Fim", cmds)
        except Exception as ex:
            print("Erro cmds", ex)

    def analyze_frames(self, name_video, num_video, child_conn, model_age, sess, model_nsfw, fn_load_image,
                       detector, timing, verbose_af=True):
        num = None
        smp = []
        num_frames = 0
        child_conn.send("processando video {:} - {:}".format(num_video, name_video))
        cap = cv2.VideoCapture(name_video)
        smp, num_frames, fps = VideoProcessor.get_samples(cap)
        if smp is None:
            cap.release()
            return {}, 0, 0, 0

        frames_video = {}
        qtd_imgs = 0
        ind = 0
        num_err = 0
        t0 = datetime.datetime.now()

        while cap.isOpened() and ind < len(smp):
            pos_frame = smp[ind]
            ind += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
            ret, frame = cap.read()
            if not ret:
                num_err += 1
                # print("erro nro {:} em frame {:} do video {:}".format(num_err, pos_frame, name_video))
                if num_err >= 15 and num_err % 15 == 0:
                    print("erro nro {:} em frame {:} do video {:}".format(num_err, pos_frame, name_video))
                if num_err > 600:
                    print("<-- analise cancelada --> nro de erros excedidos ({:}) - tentou ate frame {:} do video {:}".format(num_err, pos_frame, name_video))
                    return frames_video, qtd_imgs, num_frames, fps
                continue
            
            else:
                pass  # print("frame processado", pos_frame)

            if num_err >= 10:
                print("recuperou do {:}o erro - video {:}".format(num_err, name_video))

            num_err = 0
            qtd_imgs += 1
            # arq = os.path.join(self.files_path, f)
            t3 = datetime.datetime.now()
            # res = get_data_and_predictions(self.files_path, f, window_size, model_age,
            #                               sess, model_nsfw, fn_load_image, timing, 0.8, 0.75)
            res = VideoProcessor.get_data_and_predictions_frame(frame, pos_frame, model_age, sess, model_nsfw,
                                                      fn_load_image, detector, timing, False)
            t4 = datetime.datetime.now()
            timing["all"].append((t4 - t3).total_seconds())

            if res is not None:
                frames_video[pos_frame] = res

            # if qtd_imgs>20: break #%250==0: print(qtd_imgs)
            if qtd_imgs % 10 == 0 and verbose_af:
                    child_conn.send("{:} - {:} frames - {:} segundos".format(
                        qtd_imgs, name_video, (datetime.datetime.now() - t0).total_seconds()))

        cap.release()
        return frames_video, qtd_imgs, num_frames, fps

    def process(self, num_process, use_gpu, total_processes, queue_arqs, child_conn, num_videos, interrupt, verbose):
        # não mudar a ordem - não importar nada do keras e tf antes de configurar sessão
        from keras import backend as K
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        from keras.models import model_from_json

        config = tf.ConfigProto()
        child_conn.send(("imprime", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))

        if total_processes < 1:
            if verbose:
                child_conn.send(("imprime", "Minimo de 1 processo"))
            total_processes = 4

        if use_gpu:
            if total_processes > 4:
                if verbose:
                    child_conn.send(("imprime", "No maximo 4 total_processes com gpu"))
                total_processes = 4
            fracao = {1: 0.770, 2: 0.380, 3: 0.26, 4: 0.19}
            config.gpu_options.per_process_gpu_memory_fraction = fracao[total_processes]
        else:
            config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                    inter_op_parallelism_threads=1, allow_soft_placement=True)

        if num_process > total_processes:
            child_conn.send(("imprime", "Processo excede limite permitido"))
            child_conn.send(("imprime", "------\nAbortando {:}".format(num_process)))
            return

        if use_gpu:
            child_conn.send(("imprime", "Init p {:} de {:} na gpu, com {:.1f}% de mem. "
                  "GPU".format(num_process, total_processes, 100 * fracao[total_processes])))
        else:
            child_conn.send(("imprime", "Init p {:} de {:} na cpu".format(num_process, total_processes)))
    
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
        weights_path = "tf_open_nsfw/data/open_nsfw-weights.npy"
    
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
            child_conn.send(("imprime", "loaded models in process {:}".format(num_process)))

            timing_tmp = {}
            keys_timing = ["detect_faces", "get_faces_mtcnn2", "nsfw", "age", "all"]
            for key in keys_timing: timing_tmp[key] = []
    
            abort = False
            del_mtcnn = 0
            while not queue_arqs.empty():
                if interrupt.value == 1:
                    abort = True

                target_file_hash = None
                target_file = None
                target_hash = None
                try:
                    target_file_hash = queue_arqs.get(False)
                    target_file, target_hash = target_file_hash
                    num_videos.value += 1
                    print(num_videos.value, target_file)
                except Exception as e:
                    if not queue_arqs.empty():
                        if verbose:
                            print("imprime", "{:} queue_arqs bloqueado".format(num_process))
                        time.sleep(1)
                        continue

                    if verbose:
                        child_conn.send(("imprime", "{:} terminou via except {:}".format(num_process, e)))

                    break

                # se tiver interrompido, consome a fila sem processar
                if abort:
                    continue

                if target_file is None:
                    if verbose:
                        print("imprime", "target_file is None")
                    break


                nome_video = os.path.join(self.files_path, target_file)
                retaf = self.analyze_frames(nome_video, num_videos.value, child_conn, model_age, sess,
                                            model_nsfw, fn_load_image, detector, timing_tmp, verbose)
                frames_video, qtdimgs, num_frames, fps = retaf
    
                if qtdimgs == 0:
                    # videotmp = os.path.join(self.source_path, target_hash + ".avi") troca por causa do iped
                    videotmp = os.path.join(self.files_path, "videos_ffmpeg", target_hash + "_tmp.avi")

                    self.convert_video(nome_video, videotmp)
                    if os.path.isfile(videotmp):
                        if os.path.getsize(videotmp) > 5120:
                            retaf = self.analyze_frames(videotmp, num_videos.value, child_conn, model_age, sess,
                                                        model_nsfw, fn_load_image, detector, timing_tmp, verbose)
                            frames_video, qtdimgs, num_frames, fps = retaf
                            if qtdimgs > 0:
                                nome_video = videotmp
                        else:
                            print("arquivo pequeno")
                    else:
                        print("arquivo nao criado")
    
                frames_video["samples"] = qtdimgs
                frames_video["fps"] = fps
                frames_video["num_frames"] = num_frames
                frames_video["nomevideo"] = nome_video
                # cap.release()
                # cv2.destroyAllWindows()
                # video_data[arq] = (frames_video, target_file)
                child_conn.send(("data_file", target_file, (frames_video, target_file, timing_tmp)))
                del_mtcnn += 1
                #if del_mtcnn % 8 == 0:
                #    del detector
                #    detector = MTCNN()
                #    time.sleep(0.5)

            child_conn.send(("imprime", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
            if interrupt.value == 0:
                child_conn.send(("imprime", "Finalizou batch - processo {:}".format(num_process)))
            else:
                child_conn.send(("imprime", "Cancelando batch {:}".format(num_process)))
            try:
                child_conn.send(("finish", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")))
            except Exception as e:
                print("erro finish")

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
