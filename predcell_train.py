from predcell_subtractive_relu import *
from tensorflow import keras
import io
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/runSubtractivereluOneLayerPangram2')
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


def get_training_data(text, maxlen):
    # cut the text in semi-redundant sequences of maxlen characters
    step = 5
    sentences = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
    return sentences_to_indices_arr(sentences, maxlen)


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
maxlen = len("the quick brown fox jumps over the lazy dog ")

train_nchars = 500
test_nchars = 200

#train_text = text[:train_nchars]
train_text = ("the quick brown fox jumps over the lazy dog "*100)[:train_nchars]
x_train = get_training_data(train_text, maxlen)
x_train_onehot = to_onehot(x_train, N_CHARS)
x_train = torch.from_numpy(x_train)
x_train_onehot = torch.from_numpy(x_train_onehot)

test_text = text[train_nchars:train_nchars+test_nchars]
x_test = get_training_data(test_text, maxlen)
x_test_onehot = to_onehot(x_test, N_CHARS)
x_test = torch.from_numpy(x_test)
x_test_onehot = torch.from_numpy(x_test_onehot)

n_train_sequences = len(x_train)
n_test_sequences = len(x_test)

print(f"Training data has {len(train_text)} characters and {n_train_sequences} sequences")
print(f"Testing data has {len(test_text)} characters and {n_test_sequences} sequences")



# PredCells(num_layers, total_timesteps, hidden_dim)
num_lstms = int(input("How many stacked LSTMs to train (1 or 2)? \n"))
predcell = PredCells(num_lstms + 1, maxlen, 128, N_CHARS)

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

optimizer = torch.optim.Adam(trainable_params, lr=8e-4)
num_epochs = 3
#PATH = r'C:\Users\Samer Nour Eddine\Downloads\XAI\state_dict_model_trial.pt'
step = 0


### Periodic training setup
# Set cycle length for periodic training
cycle_length = 2
PERIODIC_TRAINING_ENABLED = True # only works if num_lstms == 2

if PERIODIC_TRAINING_ENABLED and num_lstms != 2:
    raise RuntimeError('Periodic training is meant for a 2 layer model.')

if PERIODIC_TRAINING_ENABLED:
    print(f"Periodic training enabled. Switches every {cycle_length} epochs.")

for epoch in range(num_epochs):
    if PERIODIC_TRAINING_ENABLED:
        cycle = epoch % (2 * cycle_length)
        if cycle == 0:
            # Train 2, freeze 3
            print('>> Training layer 2, freezing layer 3 <<')

            # Enable/disable the losses
            predcell.layer_losses_enabled[0] = True
            predcell.layer_losses_enabled[1] = False

            # Enable/disable training
            predcell.enable_layer_training(1)
            predcell.disable_layer_training(2)
        elif cycle == cycle_length:
            # Train 3, freeze 2
            print('>> Training layer 3, freezing layer 2 <<')

            # Enable/disable the losses
            predcell.layer_losses_enabled[0] = False
            predcell.layer_losses_enabled[1] = True

            # Enable/disable training
            predcell.disable_layer_training(1)
            predcell.enable_layer_training(2)

    # Train
    train_losses = []
    first_layer_train_losses = []
    for idx in tqdm(np.random.permutation(n_train_sequences)):
        sentence = x_train_onehot[idx]
    
        predcell.init_vars()
        loss, first_layer_loss, predictions = predcell.forward(sentence, epoch)

        # loss.retain_grad() # Is this necessary?
    
        # writer.add_graph(predcell.st_units)
        loss.backward()

        optimizer.step()

        # Putting this in makes it REALLY slow.
        # for param_name, param in names_and_params:
        #     writer.add_histogram(param_name, param, global_step=epoch)
        #     if param.grad is None:
        #         # print(f"No grad for {param_name}")
        #         pass
        #     else:
        #         writer.add_histogram(param_name+'.grad', param.grad, global_step=epoch)

        optimizer.zero_grad()
    
        train_losses.append(loss.detach().item())

        first_layer_train_losses.append(first_layer_loss.detach().item())

        

    mean_train_loss = np.mean(train_losses)
    mean_first_layer_train_loss = np.mean(first_layer_train_losses)


    # Test
    test_losses = []
    first_layer_test_losses = []
    for idx in range(n_test_sequences):
        sentence = x_test_onehot[idx]
    
        predcell.init_vars()
        loss, first_layer_loss, predictions = predcell.forward(sentence, epoch)
    
        test_losses.append(loss.detach().item())
        first_layer_test_losses.append(first_layer_loss.detach().item())

    mean_test_loss = np.mean(test_losses)
    mean_first_layer_test_loss = np.mean(first_layer_test_losses)

    writer.add_scalar('Training Loss', mean_train_loss, global_step=epoch)
    writer.add_scalar('Testing Loss', mean_test_loss, global_step=epoch)
    writer.add_scalar('Training Loss Layer 1', mean_first_layer_train_loss, global_step=epoch)
    writer.add_scalar('Testing Loss Layer 1', mean_first_layer_test_loss, global_step=epoch)
    
    # print(f"processed epoch {epoch}. loss: {mean_train_loss:.5g} train, {mean_test_loss:.5g} test; first layer loss: {mean_first_layer_train_loss:.5g} train, {mean_first_layer_test_loss:.5g} test")
    print(f"processed epoch {epoch}. Train: {mean_train_loss:.5g} Train layer1: {mean_first_layer_train_loss:.5g} Test: {mean_test_loss:.5g} Test layer1: {mean_first_layer_test_loss:.5g}")

# torch.save(predcell, "predcell_after_train_5")



def get_predictions(model, sentence):
    x = sentences_to_indices_arr([sentence], len(sentence))
    x_onehot = torch.from_numpy(to_onehot(x, N_CHARS)[0])
    loss, first_layer_loss, predictions = model(x_onehot)
    return predictions


def show_top_predictions(probs, n_top):
    chars_and_probs = [
        (INDEX_TO_CHAR[i], p.detach().item())
        for i, p in enumerate(probs)
    ]
    return sorted(chars_and_probs, key=lambda x: x[1], reverse=True)[:n_top]


input_text = "the quick brown"
predcell.init_vars()
predictions = get_predictions(predcell, input_text)
for c, probs in zip(input_text, predictions):
    print(f"Character: {c}")
    print("Predictions:")
    top_predictions = show_top_predictions(probs, 5)
    for c, p in top_predictions:
        print(f"\t{c}\t{p:.6g}")
