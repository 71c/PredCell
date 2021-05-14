''' Authors - Samer Nour Eddine (snoure01@tufts.edu), Apurva Kalia (apurva.kalia@tufts.edu)
IN PROGRESS
'''
'''
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import string
import io
import pdb
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import torchvision.models
writer = SummaryWriter(f'runs/testPredCell/tryingout_tensorboard')
# $tensorboard --logdir = "C:\Users\Samer Nour Eddine\Downloads\XAI\runs\testPredCell\tryingout_tensorboard"
# run the above command in a terminal, then navigate to http://localhost:6006 to see the Tensorboard visualization


class StateUnit(nn.Module):
    def __init__(self, layer_level, timestep, thislayer_dim, lowerlayer_dim, is_top_layer=False):
        super().__init__()
        
        self.layer_level = layer_level
        self.timestep = timestep
        self.is_top_layer = is_top_layer

        self.LSTM_ = nn.LSTM(
            input_size=thislayer_dim if is_top_layer else 2 * thislayer_dim,
            hidden_size=thislayer_dim, num_layers=1)

        self.state = torch.zeros(thislayer_dim, dtype=torch.float32)
        # reconstructions at all other time points will be determined by the state
        self.recon = torch.zeros(lowerlayer_dim, dtype=torch.float32)

        # maps from this layer to the lower layer
        # Note: includes a bias
        self.V = nn.Linear(thislayer_dim, lowerlayer_dim)

    def forward(self, BU_err, TD_err):
        self.timestep += 1
        if self.is_top_layer:
            tmp = torch.unsqueeze(BU_err, 0)
            tmp = torch.unsqueeze(tmp, 0)
            #tmp = torch.tensor(tmp, dtype = torch.float32)
            temp, _ = self.LSTM_(tmp)
            self.state = torch.squeeze(temp)
        else:
            tmp = torch.unsqueeze(torch.cat((BU_err, TD_err), axis=0), 0)
            tmp = torch.unsqueeze(tmp, 0)
            #tmp = torch.tensor(tmp, dtype = torch.float32)
            temp, _ = self.LSTM_(tmp)
            self.state = torch.squeeze(temp)
        self.recon = self.V(self.state)

    def set_state(self, input_char):
        self.state = input_char


class ErrorUnit(nn.Module):
    def __init__(self, layer_level, timestep, thislayer_dim, higherlayer_dim):
        super().__init__()
        self.layer_level = layer_level
        self.timestep = timestep
        self.TD_err = torch.squeeze(torch.tensor(np.zeros(shape=(thislayer_dim, 1)), dtype=torch.float32))
        # it shouldn't matter what we initialize this to; it will be determined by TD_err in all other iterations
        self.BU_err = torch.squeeze(torch.tensor(np.zeros(shape=(higherlayer_dim, 1)), dtype=torch.float32))
        self.W = nn.Linear(thislayer_dim, higherlayer_dim)  # maps up to the next layer

    def forward(self, state_, recon_):
        self.timestep += 1
        #self.TD_err = torch.abs(state_ - recon_)
        self.TD_err = state_ - recon_
        self.BU_err = self.W(self.TD_err.float())


class PredCells(nn.Module):
    def __init__(self, num_layers, total_timesteps, hidden_dim):
        super().__init__()
        self.num_layers = num_layers
        self.numchars = 56
        self.total_timesteps = total_timesteps
        self.st_units = []
        self.err_units = []
        self.hidden_dim = hidden_dim
        for lyr in range(self.num_layers):
            if lyr == 0:
                self.st_units.append(StateUnit(lyr, 0, self.numchars, self.numchars))
                self.err_units.append(ErrorUnit(lyr, 0, self.numchars, hidden_dim))
            elif lyr < self.num_layers - 1 and lyr > 0:
                if lyr == 1:
                    self.st_units.append(StateUnit(lyr, 0, hidden_dim, self.numchars))
                else:
                    self.st_units.append(StateUnit(lyr, 0, hidden_dim, hidden_dim))
                self.err_units.append(ErrorUnit(lyr, 0, hidden_dim, hidden_dim))
            else:
                self.st_units.append(StateUnit(lyr, 0, hidden_dim, hidden_dim, is_top_layer=True))
                self.err_units.append(ErrorUnit(lyr, 0, hidden_dim, hidden_dim))

    def forward(self, input_sentence, iternumber):
        loss = 0
        lambda1 = 0.0001
        lambda2 = 0.01
        if iternumber == 2990:
            pass
            stp = 0
        for t in range(self.total_timesteps):
            # input_char at each t is a one-hot character encoding
            input_char = input_sentence[t]  # 56 dim one hot vector
            input_char = input_char + 0.0
            input_char = torch.from_numpy(input_char)
            for lyr in range(self.num_layers):
                if lyr == 0:
                    # set the lowest state unit value to the current character
                    self.st_units[lyr].set_state(input_char)
                else:
                    self.st_units[lyr].forward(self.err_units[lyr-1].BU_err, self.err_units[lyr].TD_err)
                if lyr < self.num_layers - 1:
                    self.err_units[lyr].forward(self.st_units[lyr].state, self.st_units[lyr+1].recon)
                else:
                    pass
                if iternumber <= 1000:
                    # assign much less importance to errors at higher layers
                    loss = loss + torch.sum(torch.abs(self.err_units[lyr].TD_err))*(lambda1**(lyr))
                if iternumber > 1000:
                    # assign a bit less importance to higher layers
                    loss = loss + torch.sum(torch.abs(self.err_units[lyr].TD_err))*lambda2**(lyr)
        return loss


def init_vars(predcell):
    for lyr in range(predcell.num_layers):
        state_shape = predcell.st_units[lyr].state.shape
        recon_shape = predcell.st_units[lyr].recon.shape
        predcell.st_units[lyr].state = torch.squeeze(torch.tensor(np.zeros(shape=state_shape), dtype=torch.float32))
        predcell.st_units[lyr].recon = torch.squeeze(torch.tensor(np.zeros(shape=recon_shape), dtype=torch.float32))
        BU_shape = predcell.err_units[lyr].BU_err.shape
        predcell.err_units[lyr].BU_err = torch.squeeze(torch.tensor(np.zeros(shape=BU_shape), dtype=torch.float32))
        TD_shape = predcell.err_units[lyr].TD_err.shape
        predcell.err_units[lyr].TD_err = torch.squeeze(torch.tensor(np.zeros(shape=TD_shape), dtype=torch.float32))


path = tf.keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))
# text = text[0:50000] #reducing size to test

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 10  # 40
step = 2
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# note that this means that y[i] == x[i+1][-3]

# PredCells(num_layers, total_timesteps, hidden_dim)
PredCell = PredCells(3, maxlen, 128)
trainable_st_params = [p for model in PredCell.st_units for p in model.parameters() if p.requires_grad]
trainable_err_params = [p for model in PredCell.err_units for p in model.parameters() if p.requires_grad]


def check_st_grads(pc):
    for idx, model in enumerate(pc.st_units):
        for p in model.parameters():
            if p.grad == None:
                print("no grad found for state layer {}".format(idx))


def check_err_grads(pc):
    for idx, model in enumerate(pc.err_units):
        for p in model.parameters():
            if p.grad == None:
                print("no grad found for error layer {}".format(idx))


trainable_params = trainable_st_params + trainable_err_params

training_loss = []
optimizer = torch.optim.Adam(trainable_params)
num_epochs = 3000
stopcode = False
PATH = r'C:\Users\Samer Nour Eddine\Downloads\XAI\state_dict_model_trial.pt'
stp = False
step = 0

for epoch in range(num_epochs):
    for idx, sentence in enumerate(x[:100]):
        init_vars(PredCell)
        loss = PredCell.forward(sentence, epoch)

        loss.retain_grad()
        # writer.add_graph(PredCell.st_units)
        loss.backward()

        # check_st_grads(PredCell)
        # check_err_grads(PredCell)
        if epoch == 10:
            pass
        torch.nn.utils.clip_grad_norm(trainable_params, max_norm=1)
        optimizer.step()
        for i, param in enumerate(trainable_err_params[:-2]):
            writer.add_histogram('error_weights'+str(i), param, global_step=step)
            writer.add_histogram('error_grads'+str(i), param.grad, global_step=step)
        for i, param in enumerate(trainable_st_params[6:]):
            writer.add_histogram('st_weights'+str(6+i), param, global_step=step)
            writer.add_histogram('st_grads'+str(6+i), param.grad, global_step=step)

        #writer.add_histogram('state gradient', trainable_st_params[0].grad,global_step = step)
        optimizer.zero_grad()
        training_loss.append(loss.detach().item())
        if training_loss[-1] < 2:
            stp = True
#        if idx%1000 == 0:
#            torch.save({
#            'epoch': epoch,
#            'model_state_dict': PredCell.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': loss,
#            }, PATH)
        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1
        print("processed sentence number {} in epoch {} with loss {}".format(idx, epoch, training_loss[-1]))
debug = 0
