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
    def __init__(self, batch_size) :
        super(CNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 48, kernel_size=(1, 3), padding=(0, 3//2))
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(48, eps=0.001)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(1, 3), padding=(0, 3//2))
        self.act2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64, eps=0.001)
        self.drop2 = nn.Dropout(p=0.1)
        
        self.conv3 = nn.Conv2d(64, 100, kernel_size=(1, 3), padding=(0, 3//2))
        self.act3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(100, eps=0.001)
        self.drop3 = nn.Dropout(p=0.1)
        
        self.conv4 = nn.Conv2d(100, 150, kernel_size=(1, 7), padding=(0, 7//2))
        self.act4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(150, eps=0.001)
        self.drop4 = nn.Dropout(p=0.1)
        
        self.conv5 = nn.Conv2d(150, 300, kernel_size=(1, 7), padding=(0, 7//2))
        self.act5 = nn.ReLU()
        self.norm5 = nn.BatchNorm2d(300, eps=0.001)
        self.drop5 = nn.Dropout(p=0.1)
        
        self.maxpool_5 = nn.MaxPool2d((1, 3))
        
        self.conv6 = nn.Conv2d(300, 200, kernel_size=(1, 7), padding=(0, 7//2))
        self.act6 = nn.ReLU()
        self.norm6 = nn.BatchNorm2d(200, eps=0.001)
        self.drop6 = nn.Dropout(p=0.1)
        
        self.conv7 = nn.Conv2d(200, 200, kernel_size=(1, 3), padding=(0, 3//2))
        self.act7 = nn.ReLU()
        self.norm7 = nn.BatchNorm2d(200, eps=0.001)
        self.drop7 = nn.Dropout(p=0.1)
        
        self.conv8 = nn.Conv2d(200, 200, kernel_size=(1, 3), padding=(0, 3//2))
        self.act8 = nn.ReLU()
        self.norm8 = nn.BatchNorm2d(200, eps=0.001)
        self.drop8 = nn.Dropout(p=0.1)
        
        self.maxpool_8 = nn.MaxPool2d((1, 4))
        
        self.conv9 = nn.Conv2d(200, 200, kernel_size=(1, 7), padding=(0, 7//2))
        self.act9 = nn.ReLU()
        self.norm9 = nn.BatchNorm2d(200, eps=0.001)
        self.drop9 = nn.Dropout(p=0.1)
        
        self.maxpool_9 = nn.MaxPool2d((1, 4))
        
        self.fc10 = nn.Linear(in_features=600, out_features=100)
        self.act10 = nn.ReLU()
        self.norm10 = nn.BatchNorm1d(100, eps=0.001)
        self.drop10 = nn.Dropout(p=0.1)
        
        self.fc11 = nn.Linear(in_features=100, out_features=12)
        
        self.batch_size = batch_size
        self.use_cuda = True if torch.cuda.is_available() else False
        
    def forward(self, x):
        
        x = self.drop1(self.norm1(self.act1(self.conv1(x))))
        x = self.drop2(self.norm2(self.act2(self.conv2(x))))
        x = self.drop3(self.norm3(self.act3(self.conv3(x))))
        x = self.drop4(self.norm4(self.act4(self.conv4(x))))
        x = self.maxpool_5(self.drop5(self.norm5(self.act5(self.conv5(x)))))
        
        x = self.drop6(self.norm6(self.act6(self.conv6(x))))
        x = self.drop7(self.norm7(self.act7(self.conv7(x))))
        x = self.maxpool_8(self.drop8(self.norm8(self.act8(self.conv8(x)))))
        
        x = self.maxpool_9(self.drop9(self.norm9(self.act9(self.conv9(x)))))
        
        #x = x.view(-1, 600)
        x = x.transpose(1, 3)
        x = x.reshape(-1, 600)
        
        x = self.drop10(self.norm10(self.act10(self.fc10(x))))
        y = self.fc11(x)[:, 5].unsqueeze(-1)
        
        return y
        

def indexes_from_sentence(lang, sentence):
    return [lang.token2index[t] for t in sentence]

class DragoNNClassifier():
    def __init__(self, seq_len=145, batch_size=64, learning_rate=0.001, epochs=50, run_name='mpradragonn_pytorch'):
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
