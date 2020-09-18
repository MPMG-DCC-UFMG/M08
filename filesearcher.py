import os
import sys
import datetime
import time
import hashlib
import string
import math
from shutil import rmtree

def imprime_msg(signal_msg, task_id, txt):
    if signal_msg is not None:
        signal_msg.send( ('imprime',  txt) )
        
def calcula_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_arquivos_hashes_subdir(root_dir, extensions, min_size=4096, verbose_gf=False, signal_msg=None,
                               task_id=""):
    if verbose_gf:
        imprime_msg(signal_msg, task_id, "Calcula hashs dos arquivos {:}".format(extensions))
        imprime_msg(signal_msg, task_id, " Diretório root_dir {:}".format(root_dir))
        imprime_msg(signal_msg, task_id, " Tamanho minimo {:}".format(min_size))
    num = 0
    nomes = []
    hashes = {}

    for r,d,f in os.walk(root_dir):
        dir_rel = r[len(root_dir):]
#         if case_type == Case.TIPO_DIRECTORY:
#             if dir_rel.lower().startswith("framesvid") or dir_rel.lower().startswith("relatorio"):
#                 continue

        for arquivo_rel in f:
            arquivo_abs = os.path.join(r, arquivo_rel)
            sz = os.path.getsize(arquivo_abs)
            arquivo_rel = arquivo_abs[len(root_dir):]
            ind = arquivo_rel.rfind(".")

            if ind > 0 and os.path.isfile(arquivo_abs) and sz >= min_size:
                ext = arquivo_rel[ind:]
                if ext.lower() in extensions:
                    md5 = calcula_md5(arquivo_abs)
                    try:
                        nomes.append(arquivo_rel)
                        if md5 not in hashes:
                            hashes[md5] = []
                        hashes[md5].append(arquivo_rel)
                        num += 1
                        if verbose_gf and num % 200 == 0:
                            imprime_msg(signal_msg, task_id, "    --> {:}".format(num))
                            # break
                    except:
                        imprime_msg(signal_msg, task_id, "  Erro : {:}".format(arquivo_abs))

    if verbose_gf:
        imprime_msg(signal_msg, task_id, "    --> {:}".format(num))

    return nomes, hashes


def get_arquivos_hashes_100k_subdir(root_dir, extensions, min_size=5120, verbose_gf=False, signal_msg=None,
                                    task_id=""):
    if verbose_gf:
        imprime_msg(signal_msg, task_id, "Calcula hashs dos arquivos hashs {:}".format(extensions))
        imprime_msg(signal_msg, task_id, " Diretório root_dir {:}".format(root_dir))
        imprime_msg(signal_msg, task_id, " Tamanho minimo {:}".format(min_size))

    num = 0
    nomes = []
    hashes = {}

    for r, d, f in os.walk(root_dir):
        dir_rel = r[len(root_dir):]
#         if case_type == Case.TIPO_DIRECTORY:
#             if dir_rel.lower().startswith("framesvid") or dir_rel.lower().startswith("relatorio"):
#                 continue

        for arquivo_rel in f:
            arquivo_abs = os.path.join(r, arquivo_rel)
            sz = os.path.getsize(arquivo_abs)
            arquivo_rel = arquivo_abs[len(root_dir):]
            ind = arquivo_rel.rfind(".")

            if ind > 0 and os.path.isfile(arquivo_abs) and sz >= min_size:
                ext = arquivo_rel[ind:]
                if ext.lower() in extensions:
                    # md5 = calcula_md5_100k(arquivo_abs, sz)
                    md5 = calcula_md5(arquivo_abs)

                    try:
                        nomes.append(arquivo_rel)
                        if md5 not in hashes:
                            hashes[md5] = []
                        hashes[md5].append(arquivo_rel)
                        num += 1
                        if verbose_gf and num % 80 == 0:
                            imprime_msg(signal_msg, task_id, "    --> {:}".format(num))
                        #if num % 200 == 0:
                        #    break
                    except:
                        imprime_msg(signal_msg, task_id, "  Erro : {:}".format(arquivo_abs))

    if verbose_gf:
        imprime_msg(signal_msg, task_id, "    --> {:}".format(num))

    return nomes, hashes


class FileSearcher:
    """ Class to extract files from directory or report """
    valido = string.ascii_letters + string.digits + "_"
    def __init__(self, path):
        # path must end with "/"
        self.task_id = ""
        self.source_path = path
        self.files_path = path
        self.subdir = ""
        self.img_ext = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        self.vid_ext = ['.wav', '.qt', '.mpeg', '.mpg', '.avi', '.mp4', '.3gp', '.mov', '.lvl', '.m4v', '.wmv']
        self.img_query = 'categoria:("Outras Imagens" OR "Imagens Temporárias Internet" OR' \
                         ' "Imagens em Pasta de Sistema" OR "Possíveis Digitalizações") '
        self.vid_query = 'categoria:("Vídeos") '
        self.min_img_size = 5*1024
        self.min_vid_size = 50*1024
        self.verbose_fs = False
        self.img_names = self.img_ids = self.img_hashes = self.img_itens_list = self.img_files_list = None
        self.vid_names = self.vid_ids = self.vid_hashes = self.vid_itens_list = self.vid_files_list = None
        self.files = {"images": {}, "videos": {}}
        self.err_message = None

    def get_from_directory(self, signal_msg, task_id, min_img_size=5*1024, min_vid_size=100*1024, 
                           subdir="", verbose_fs=False):
        
        self.task_id = task_id
        self.files_path = self.source_path + subdir
        self.subdir = subdir
        self.min_img_size = min_img_size
        self.min_vid_size = min_vid_size
        self.verbose_fs = verbose_fs
        signal_msg.send(('imprime', '{}: {} - {}'.format('Inicia busca por imagens em', self.source_path, 
                                             datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) 
                        ))
        
        # --- imagens ----
        img_names, img_ids = \
            get_arquivos_hashes_subdir(self.files_path, self.img_ext, min_size=self.min_img_size, verbose_gf=verbose_fs,
                                       signal_msg=signal_msg, task_id=self.task_id)
        # img_hashes = list(img_ids.keys())
        self.img_itens_list = list(img_ids.items())
        self.img_files_list = [f[1][0] for f in self.img_itens_list]
        i = 0
        for i in range(0, len(self.img_itens_list)):
            file_names = self.img_itens_list[i][1]
            file_name = file_names[0]
            file_hash = self.img_itens_list[i][0]
            file_id = "{:07d}".format(i)
            self.files["images"][os.path.join(self.source_path,file_name)] = {"names": file_names, "hash": file_hash, "id": file_id}
        last_i = i
        signal_msg.send(('imprime',  
                         '{}: {}\n'.format("Número de arquivos de imagem únicos",len(self.img_files_list))
                        ))

        signal_msg.send(('imprime', '{}: {} - {}'.format('Inicia busca por vídeos em', self.source_path, 
                                             datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) 
                        ))
        
        # --- videos ----
        vid_names, vid_ids = \
            get_arquivos_hashes_100k_subdir(self.files_path, self.vid_ext, min_size=self.min_vid_size,
                                            verbose_gf=verbose_fs, signal_msg=signal_msg,
                                            task_id=self.task_id)
        
        # vid_hashes = list(self.vid_ids.keys())
        self.vid_itens_list = list(vid_ids.items())
        self.vid_files_list = [f[1][0] for f in self.vid_itens_list]
        for i in range(0, len(self.vid_itens_list)):
            file_names = self.vid_itens_list[i][1]
            file_name = file_names[0]
            file_hash = self.vid_itens_list[i][0]
            file_id = "{:07d}".format(i + last_i)
            self.files["videos"][os.path.join(self.source_path,file_name)] = {"names": file_names, "hash": file_hash, "id": file_id}

        signal_msg.send(('imprime', 
                         '{}: {}\n'.format("Número de arquivos de vídeo únicos",len(self.vid_files_list))
                       ))

        signal_msg.send(('imprime', 
                        '{} - {}'.format('Finalizado', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                        ))
        return
