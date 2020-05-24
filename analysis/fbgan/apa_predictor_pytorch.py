#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import random, os, h5py, math, time, glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from utils.utils import *
from utils.bio_utils import *
from utils.lang_utils import *

class CNNClassifier(nn.Module):
    def __init__(self, batch_size, lib_index=5, distal_pas=1.) :
        super(CNNClassifier, self).__init__()
        
        lib_inp_numpy = np.zeros((batch_size, 13))
        lib_inp_numpy[:, lib_index] = 1.
        self.lib_inp = Variable(torch.FloatTensor(lib_inp_numpy).to(torch.device('cuda:0')))
        
        d_inp_numpy = np.zeros((batch_size, 1))
        d_inp_numpy[:, 0] = distal_pas
        self.d_inp = Variable(torch.FloatTensor(d_inp_numpy).to(torch.device('cuda:0')))
        
        self.conv1 = nn.Conv2d(4, 96, kernel_size=(1, 8))
        self.maxpool_1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(96, 128, kernel_size=(1, 6))
        
        self.fc1 = nn.Linear(in_features=94 * 128 + 1, out_features=256)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=256 + 13, out_features=1)
        
        self.batch_size = batch_size
        self.use_cuda = True if torch.cuda.is_available() else False
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv2(x))
        
        x = x.transpose(1, 3)
        x = x.reshape(-1, 94 * 128)
        
        x = torch.cat([x, self.d_inp], dim=1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = torch.cat([x, self.lib_inp], dim=1)
        
        x = F.sigmoid(self.fc2(x))
        
        return x
        

def indexes_from_sentence(lang, sentence):
    return [lang.token2index[t] for t in sentence]

class APAClassifier():
    def __init__(self, seq_len=205, batch_size=64, learning_rate=0.001, epochs=50, run_name='aparent_pytorch'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_epochs = epochs
        self.learning_rate = learning_rate
        self.use_gpu = True if torch.cuda.is_available() else False
        
        self.build_model()
        self.checkpoint_dir = './checkpoint/' + run_name + '/'
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.init_epoch = self.load_model()

    def build_model(self):
        self.cnn = CNNClassifier(self.batch_size)
        if self.use_gpu:
            self.cnn.cuda()
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

    def save_model(self, epoch):
        torch.save(self.cnn.state_dict(), self.checkpoint_dir + "model_weights_{}.pth".format(epoch))

    def load_model(self):
        '''
            Load model parameters from most recent epoch
        '''
        list_model = glob.glob(self.checkpoint_dir + "model*.pth")
        if len(list_model) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        chk_file = max(list_model, key=os.path.getctime)
        epoch_found = int( (chk_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found!".format(epoch_found))
        self.cnn.load_state_dict(torch.load(chk_file))
        return epoch_found

    def onehot_encode_seqs(self, seqs):
        input_onehot = np.zeros((len(seqs), 4, 1, self.seq_len))
        for i in range(0, len(seqs)) :
            for j in range(0, len(seqs[i])) :
                if seqs[i][j] == 'A' :
                    input_onehot[i, 0, 0, j] = 1.
                elif seqs[i][j] == 'C' :
                    input_onehot[i, 1, 0, j] = 1.
                elif seqs[i][j] == 'G' :
                    input_onehot[i, 2, 0, j] = 1.
                elif seqs[i][j] == 'T' :
                    input_onehot[i, 3, 0, j] = 1.
        
        return input_onehot
    
    def predict_model(self, input_seqs):
        pos_seqs = []
        
        self.cnn.eval()
        
        num_pred_batches = int(len(input_seqs)/self.batch_size)
        all_preds = np.zeros((num_pred_batches*self.batch_size, 1))
        for idx in range(num_pred_batches):
            batch_seqs = input_seqs[idx*self.batch_size:(idx+1)*self.batch_size]
            
            input_onehot = self.onehot_encode_seqs(batch_seqs)

            #input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
            input_var = Variable(torch.FloatTensor(input_onehot))
            
            input_var = input_var.cuda() if self.use_gpu else input_var
            y_pred = self.cnn(input_var)
            #print( "Made predictions...")
            all_preds[idx*self.batch_size:(idx+1)*self.batch_size,:] = y_pred.data.cpu().numpy()
        return all_preds

if __name__ == '__main__':
    main()
