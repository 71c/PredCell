from predcell import *
import tensorflow as tf
import io
import numpy as np

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/run8')
# $tensorboard --logdir "runs"
# run something like the above command in a terminal, then navigate to http://localhost:6006 to see the Tensorboard visualization
# have a different run folder for different runs of your program

path = tf.keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))
# text = text[0:50000] #reducing size to test

chars = sorted(list(set(text)))
n_chars = len(chars)
print("Total chars:", n_chars)
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

x = np.zeros((len(sentences), maxlen, n_chars), dtype=np.bool)
y = np.zeros((len(sentences), n_chars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# note that this means that y[i] == x[i+1][-3]

# PredCells(num_layers, total_timesteps, hidden_dim)
predcell = PredCells(3, maxlen, 128, n_chars)

trainable_st_params = [p for model in predcell.st_units for p in model.parameters() if p.requires_grad]
trainable_err_params = [p for model in predcell.err_units for p in model.parameters() if p.requires_grad]


names_and_params = []
for lyr, (st_unit, err_unit) in enumerate(zip(predcell.st_units, predcell.err_units)):
    names_and_params.append((f'st_units[{lyr}].V.weight', st_unit.V.weight))
    names_and_params.append((f'st_units[{lyr}].V.bias', st_unit.V.bias))

    if type(st_unit.LSTM_) is torch.nn.modules.rnn.LSTM:
        names_and_params.append((f'st_units[{lyr}].LSTM.weight_ih_l', st_unit.LSTM_.weight_ih_l0))
        names_and_params.append((f'st_units[{lyr}].LSTM.weight_hh_l', st_unit.LSTM_.weight_hh_l0))
        names_and_params.append((f'st_units[{lyr}].LSTM.bias_ih_l', st_unit.LSTM_.bias_ih_l0))
        names_and_params.append((f'st_units[{lyr}].LSTM.bias_hh_l', st_unit.LSTM_.bias_hh_l0))
    elif type(st_unit.LSTM_) is torch.nn.modules.rnn.LSTMCell:
        names_and_params.append((f'st_units[{lyr}].LSTM.weight_ih', st_unit.LSTM_.weight_ih))
        names_and_params.append((f'st_units[{lyr}].LSTM.weight_hh', st_unit.LSTM_.weight_hh))
        names_and_params.append((f'st_units[{lyr}].LSTM.bias_ih', st_unit.LSTM_.bias_ih))
        names_and_params.append((f'st_units[{lyr}].LSTM.bias_hh', st_unit.LSTM_.bias_hh))

    names_and_params.append((f'err_units[{lyr}].W.weight', err_unit.W.weight))
    names_and_params.append((f'err_units[{lyr}].W.bias', err_unit.W.bias))


# print(list(zip(*names_and_params))[0])


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

# print(trainable_st_params[0])
# print(predcell.st_units[0].V.weight)

training_loss = []
optimizer = torch.optim.Adam(trainable_params)
num_epochs = 5
PATH = r'C:\Users\Samer Nour Eddine\Downloads\XAI\state_dict_model_trial.pt'
stp = False
step = 0

for epoch in range(num_epochs):
    for idx, sentence in enumerate(x[:100]):
        # init_vars(predcell)
        predcell.init_vars()
        loss = predcell.forward(sentence, epoch)

        loss.retain_grad()
        # writer.add_graph(predcell.st_units)
        loss.backward()

        # check_st_grads(predcell)
        # check_err_grads(predcell)
        if epoch == 10:
            pass
        torch.nn.utils.clip_grad_norm(trainable_params, max_norm=1)
        optimizer.step()

        for param_name, param in names_and_params:
            writer.add_histogram(param_name, param, global_step=step)
            if param.grad is None:
                # print(f"No grad for {param_name}")
                pass
            else:
                writer.add_histogram(param_name+'.grad', param.grad, global_step=step)

        # for i, param in enumerate(trainable_err_params[:-2]):
        #     writer.add_histogram('error_weights'+str(i), param, global_step=step)
        #     writer.add_histogram('error_grads'+str(i), param.grad, global_step=step)
        # for i, param in enumerate(trainable_st_params[6:]):
        #     writer.add_histogram('st_weights'+str(6+i), param, global_step=step)
        #     writer.add_histogram('st_grads'+str(6+i), param.grad, global_step=step)

        optimizer.zero_grad()
        training_loss.append(loss.detach().item())
        if training_loss[-1] < 2: 
            stp = True           ##### What are you doing here??? #######

        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1
        print("processed sentence number {} in epoch {} with loss {}".format(idx, epoch, training_loss[-1]))
debug = 0
