from predcell import *
from tensorflow import keras
import io
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/run15')
# $tensorboard --logdir "runs"
# run something like the above command in a terminal, then navigate to http://localhost:6006 to see the Tensorboard visualization
# have a different run folder for different runs of your program

CHAR_TO_INDEX = {}
INDEX_TO_CHAR = {}

def get_nietzsche_text():
    path = keras.utils.get_file(
        "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
    )
    with io.open(path, encoding="utf-8") as f:
        text = f.read().lower()
    text = text.replace("\n", " ")  # We remove newlines chars for nicer display
    print("Corpus length:", len(text))
    return text


def sentences_to_indices_arr(sentences, maxlen):
    x = np.zeros((len(sentences), maxlen), dtype=np.int64)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t] = CHAR_TO_INDEX[char]
    return x


def get_training_data(text, maxlen, n_examples=None):
    # cut the text in semi-redundant sequences of maxlen characters
    step = 5
    sentences = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
    n_examples = len(sentences) if n_examples is None else n_examples
    print("Number of sequences:", n_examples)
    return sentences_to_indices_arr(sentences[:n_examples], maxlen)


def to_onehot(data, n_items=None):
    '''data: ndarray of integers beginning with 0'''
    n_items = data.max() + 1 if n_items is None else n_items
    ret = np.zeros(data.shape + (n_items,), dtype=np.float32)
    it = np.nditer(data, flags=['multi_index'])
    for val in it:
        ret[it.multi_index + (val,)] = 1
    return ret


### Get text data and set up stuff
text = get_nietzsche_text()
chars = sorted(list(set(text)))
N_CHARS = len(chars)
CHAR_TO_INDEX = {c: i for i, c in enumerate(chars)}
INDEX_TO_CHAR = {i: c for i, c in enumerate(chars)}
print("Total chars:", N_CHARS)

### Get training data
maxlen = 10
n_examples = 100
x = get_training_data(text, maxlen, n_examples)
x_onehot = to_onehot(x, N_CHARS)

x = torch.from_numpy(x)
x_onehot = torch.from_numpy(x_onehot)


# PredCells(num_layers, total_timesteps, hidden_dim)
predcell = PredCells(3, maxlen, 128, N_CHARS)

# predcell = torch.load("predcell_after_train_3")


trainable_st_params = [p for model in predcell.st_units for p in model.parameters() if p.requires_grad]
trainable_err_params = [p for model in predcell.err_units for p in model.parameters() if p.requires_grad]

# Get all the parameters along with their associated names.
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


trainable_params = trainable_st_params + trainable_err_params

optimizer = torch.optim.Adam(trainable_params, lr=0.0008)
num_epochs = 7
PATH = r'C:\Users\Samer Nour Eddine\Downloads\XAI\state_dict_model_trial.pt'
stp = False
step = 0



for epoch in range(num_epochs):
    losses = []
    first_layer_losses = []
    for idx in tqdm(np.random.permutation(n_examples)):
        sentence = x_onehot[idx]
    
        predcell.init_vars()
        loss, first_layer_loss, predictions = predcell.forward(sentence, epoch)

        loss.retain_grad()
        # writer.add_graph(predcell.st_units)
        loss.backward()

        if epoch == 10:
            pass
        # torch.nn.utils.clip_grad_norm(trainable_params, max_norm=1)
        optimizer.step()

        for param_name, param in names_and_params:
            writer.add_histogram(param_name, param, global_step=step)
            if param.grad is None:
                # print(f"No grad for {param_name}")
                pass
            else:
                writer.add_histogram(param_name+'.grad', param.grad, global_step=step)

        optimizer.zero_grad()
    
        losses.append(loss.detach().item())

        first_layer_losses.append(first_layer_loss.detach().item())

    mean_loss = np.mean(losses)
    mean_first_layer_loss = np.mean(first_layer_losses)
    writer.add_scalar('Training Loss', mean_loss, global_step=epoch)
    
    print("processed epoch {} with loss {}, first layer loss {}".format(epoch, mean_loss, mean_first_layer_loss))

# torch.save(predcell, "predcell_after_train_5")




# sentence = x_onehot[0]
# predcell.init_vars()
# predcell.forward(sentence, 2000)
# print(predcell.st_units[1].recon)
# print(predcell.st_units[0].state)


def get_predictions(model, sentence):
    x = sentences_to_indices_arr([sentence], len(sentence))
    x_onehot = torch.from_numpy(to_onehot(x, N_CHARS)[0])
    loss, first_layer_loss, predictions = model(x_onehot, 2000)
    return predictions


def show_top_predictions(probs, n_top):
    chars_and_probs = [
        (INDEX_TO_CHAR[i], p.detach().item())
        for i, p in enumerate(probs)
    ]
    return sorted(chars_and_probs, key=lambda x: x[1], reverse=True)[:n_top]


input_text = "this is a test"
predictions = get_predictions(predcell, input_text)
for c, probs in zip(input_text, predictions):
    print(f"Character: {c}")
    print("Predictions:")
    top_predictions = show_top_predictions(probs, 5)
    for c, p in top_predictions:
        print(f"\t{c}\t{p:.6g}")
