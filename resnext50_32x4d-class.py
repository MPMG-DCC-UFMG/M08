import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from mtcnn.mtcnn import MTCNN
from faces import get_faces_mtcnn
from PIL import Image
import sys, cv2, os
import numpy as np
import pandas as pd
import shutil

args = {'batch_size': 10}

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')
    
print(args['device'])

############################## CROP FACES ##############################
print('Detectando faces')
try:
    files_path = sys.argv[1]
except:
    print('Processo precisa do argumento files_path com a localizacao dos arquivos para análise')
    exit(1)
    
    
def load_img(filename):
    if not os.path.isfile(filename):
        print("not os.path.isfile({:})".format(filename))
        return None, None
    
    try:
        img = cv2.imread(filename)
    except Exception as ex:
        print("Erro em cv2.imread: {:}:{:}".format(filename, ex))
        return None, None
        
    if img is None:
        print("img_work=None em cv2.imread: {:}".format(filename))
        return None, None
    
    shape_img = img.shape
    min_dim, max_dim = min(shape_img[0], shape_img[1]), max(shape_img[0], shape_img[1])
    if min_dim < 30 or max_dim < 60:
        print("image is too small")
        return None, None
    
    lim_inferior, lim_superior = 720, 1440
    if not ( (max_dim > lim_inferior) and (max_dim < lim_superior) ):
        scale_factor = lim_inferior if (lim_inferior/max_dim) > 1 else lim_superior
        scale_factor /= max_dim
        new_shape_img = (int(shape_img[0] * scale_factor),
                         int(shape_img[1] * scale_factor) )
        print('{:} rescaled by {:.2f}. Original size: {:}, New size: {:}'.format(filename,scale_factor, shape_img, new_shape_img))
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), new_shape_img
    
    
timing = {"detect_faces": [], "nsfw": [], "age": [], "all": []}
# # TODO: DELETE FOLDER AT THE END
save_cropped_path = os.path.join(os.getcwd(), 'tmp_faces')
if not os.path.isdir( save_cropped_path ):
    os.mkdir(save_cropped_path)

detector = MTCNN()
for file in os.listdir(files_path):
    
    filename = os.path.join(files_path, file)
    
    img, shape_img = load_img(filename)
    try:
        faces = get_faces_mtcnn(img, detector, timing["detect_faces"])
    except Exception as ex:
        print('Erro detectando as faces:', ex)
        continue
    
    for k, face in enumerate(faces):
        face_img, coord, confid_face = face
        area_face = (coord[2] - coord[0]) * (coord[3] - coord[1])
        prop_x = (coord[3] - coord[1]) / shape_img[1]
        if area_face < 1200 and prop_x < 0.06:
            continue
        cv2.imwrite( os.path.join(save_cropped_path, str(k)+'_'+file) , cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  )
        
del detector
# ########################################################################


############################ AGE INFERENCE #############################

print('Rodando modelo de inferência de idade')

ages = [np.arange(0,3), np.arange(4,7), np.arange(8, 14), np.arange(15, 21),
        np.arange(25, 33), np.arange(38, 44), np.arange(48, 54), np.arange(60, 121)]

def age2class(age):
    for k, a in enumerate(ages):
        if age <= a[-1]: return k

### CLASSIFICATION NET 
class NET(nn.Module):
    
    def __init__(self, num_classes, gender=False):
        
        super(NET, self).__init__()
        
        resnext = models.resnext50_32x4d(pretrained=True)
        self.base1 = nn.Sequential(*list(resnext.children())[:5])
        self.base2 = nn.Sequential(*list(resnext.children())[5:-1])
        
        self.age_stream    = nn.Sequential(
                                nn.Linear(2048, 512),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(512, num_classes)
                            )
        self.isgender = gender
        if self.isgender:
            self.gender_stream = nn.Sequential(
                                    nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 1),
                                    nn.Sigmoid()
                                )
        
    def forward(self, X):
        
        feature = self.base1(X) 
        feature = self.base2(feature) 
        feature = feature.view(feature.size(0), -1)
        age     = self.age_stream(feature)
        
        if self.isgender:
            gender  = self.gender_stream(feature) 
            return age, gender
        
        return age


net = NET(len(ages)).to(args['device'])
net.load_state_dict(torch.load( os.path.join(os.getcwd(), 'dados_cnn', 'model_Adience0.pt') ))

### LOADING TEST DATA
class Dados(Dataset):
    
    def __init__(self, root_path):
        self.filenames = []
        for root, dirs, files in os.walk(root_path):
            print(root)
            self.filenames.extend([os.path.join(root, f) for f in files])
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        img = Image.open(self.filenames[idx])
        img = self.transform(img)
        
        return self.filenames[idx], img 
    
dados = Dados(os.path.join(save_cropped_path))
loader = DataLoader(dados, batch_size=args['batch_size'])

save_data = {'filepaths': [], 'class_estimation': [], 
            'argsort_estimation': []}
net.eval()
for k, batch in enumerate(loader):
    print('\r{0}/{1}'.format(k, len(loader)), end="", flush=True)
    with torch.no_grad():
        path, dado = batch
        
        # Cast do dado na GPU
        dado = dado.to(args['device'])

        # Forward
        ypred = net(dado)
        _, pred = torch.max(ypred, axis=-1)
        pred = pred.detach().cpu().data.numpy()
            
        save_data['filepaths'].extend(list(path))
        save_data['class_estimation'].extend(list( pred ))
        save_data['argsort_estimation'].extend( list(torch.argsort(ypred.detach(), descending=True).cpu().numpy()) )

  
pd.DataFrame.from_dict(save_data).to_csv('results.txt')
print('\nDone\n')
####################################################################
