import torch
import torch.nn as nn
import numpy as np

class StateUnit(nn.Module):
    def __init__(self, layer_level, thislayer_dim, lowerlayer_dim, is_top_layer=False):
        super().__init__()
        
        self.layer_level = layer_level
        self.is_top_layer = is_top_layer        
        self.thislayer_dim = thislayer_dim
        self.lowerlayer_dim = lowerlayer_dim

        # Set state, recon, timestep to 0
        self.init_vars()

        # maps from this layer to the lower layer
        # Note: includes a bias
        self.V = nn.Linear(thislayer_dim, lowerlayer_dim)
        
        self.LSTM_ = nn.LSTMCell(
            input_size=thislayer_dim if is_top_layer else 3 * thislayer_dim,
            hidden_size=thislayer_dim)
        
        # Add extra attribute to each parameter for use in enabling/disabling training
        for p in self.parameters():
            p.initially_requires_grad = p.requires_grad

    def forward(self, BU_err, TD_err):
        self.timestep += 1

        if self.is_top_layer:
            input = torch.unsqueeze(BU_err, 0) # make it so there is 1 batch
        else:
            input = torch.unsqueeze(torch.cat((BU_err, TD_err), axis=0), 0)
        h_0 = torch.unsqueeze(self.state, 0) # make 1 batch
        c_0 = torch.unsqueeze(self.cell_state, 0) # make 1 batch
        h_1, c_1 = self.LSTM_(input, (h_0, c_0))
        self.state = torch.squeeze(h_1) # remove batch
        self.cell_state = torch.squeeze(c_1) # remove batch

        self.recon = self.V(self.state)

        # if self.layer_level == 1:
        #     self.recon = nn.functional.softmax(self.recon)

    def set_state(self, input_char):
        self.state = (input_char - torch.mean(input_char))/torch.std(input_char)

    def init_vars(self):
        '''Sets state and reconstruction to zero, and set timestep to 0'''
        self.timestep = 0
        self.state = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        self.cell_state = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        # reconstructions at all other time points will be determined by the state
        self.recon = torch.zeros(self.lowerlayer_dim, dtype=torch.float32)
    
    def enable_training(self):
        for p in self.parameters():
            if p.initially_requires_grad:
                p.requires_grad = True
    
    def disable_training(self):
        for p in self.parameters():
            if p.initially_requires_grad:
                p.requires_grad = False


class ErrorUnit(nn.Module):
    def __init__(self, layer_level, thislayer_dim, higherlayer_dim):
        super().__init__()
        self.layer_level = layer_level
        self.thislayer_dim = thislayer_dim
        self.higherlayer_dim = higherlayer_dim

        self.init_vars()

        self.W = nn.Linear(thislayer_dim*2, higherlayer_dim)  # maps up to the next layer
        self.relu = nn.ReLU()

        # Add extra attribute to each parameter for use in enabling/disabling training
        for p in self.parameters():
            p.initially_requires_grad = p.requires_grad

    def forward(self, state, recon):
        self.timestep += 1
        #self.TD_err = torch.abs(state - recon)
        self.TD_err = torch.cat((self.relu(state - recon),self.relu(recon - state)))
        self.BU_err = self.W(self.TD_err.float())
    
    def init_vars(self):
        '''Sets TD_err and BU_err to zero, and set timestep to 0'''
        self.timestep = 0
        self.TD_err = torch.zeros(self.thislayer_dim*2, dtype=torch.float32)
        # it shouldn't matter what we initialize this to; it will be determined by TD_err in all other iterations
        self.BU_err = torch.zeros(self.higherlayer_dim, dtype=torch.float32)
    
    def enable_training(self):
        for p in self.parameters():
            if p.initially_requires_grad:
                p.requires_grad = True
    
    def disable_training(self):
        for p in self.parameters():
            if p.initially_requires_grad:
                p.requires_grad = False


class PredCells(nn.Module):
    def __init__(self, num_layers, total_timesteps, hidden_dim, numchars):
        super().__init__()
        self.num_layers = num_layers
        self.numchars = numchars
        self.total_timesteps = total_timesteps
        self.st_units = []
        self.err_units = []
        self.hidden_dim = hidden_dim
        for lyr in range(self.num_layers):
            if lyr == 0:
                self.st_units.append(StateUnit(lyr, numchars, numchars))
                self.err_units.append(ErrorUnit(lyr, numchars, hidden_dim))
            elif lyr == self.num_layers - 1:
                self.st_units.append(StateUnit(lyr, hidden_dim, hidden_dim if lyr != 1 else numchars, is_top_layer=True))
                self.err_units.append(ErrorUnit(lyr, hidden_dim, hidden_dim))
            else:
                if lyr == 1:
                    self.st_units.append(StateUnit(lyr, hidden_dim, numchars))
                else:
                    self.st_units.append(StateUnit(lyr, hidden_dim, hidden_dim))
                self.err_units.append(ErrorUnit(lyr, hidden_dim, hidden_dim))
        
        # Element number i indicates whether the TD error of the i'th ErrorUnit contributes to the loss function.
        # There are self.num_layers - 1, rather than self.num_layers, items in this list because the top/last layer's TD error is always 0s';
        # basically it doesn't really have a TD error.
        self.layer_losses_enabled = [True for lyr in range(self.num_layers - 1)]

    def forward(self, input_sentence, iternumber= 1e10):
        loss = 0
        first_layer_loss = 0
        predictions = []

        # print(self.err_units[self.num_layers - 1].W.weight) # This should be constant.

        for t in range(min(self.total_timesteps, len(input_sentence))):
            # input_char at each t is a one-hot character encoding
            input_char = input_sentence[t]  # 56 dim one hot vector
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
                
                # Update Loss
                #lambda1 = (np.cos(x)+ x)/(x + 1) if lyr == 0 else (.5*np.sin(x) + x/7)/(.5 + x/7)
                lambda1 = 1.0 if lyr ==0 else 0.75#(iternumber%10)/10
                
                if lyr != self.num_layers - 1:
                    # If lyr == self.num_layers - 1, i.e. this is the top layer,
                    # then the TD error at this layer will be 0 so it does not make a difference if we include this condition.
                    if self.layer_losses_enabled[lyr]:
                        loss += torch.sum(torch.abs(self.err_units[lyr].TD_err))*lambda1

                # if lyr == 0:
                #     loss += torch.sum(torch.abs(self.err_units[lyr].TD_err))

                # We can also do it in the simple manner specified on the powerpoint
                # loss += torch.sum(torch.abs(self.err_units[lyr].TD_err))
            
            first_layer_loss += torch.sum(torch.abs(self.err_units[0].TD_err))
            predictions.append(self.st_units[1].recon)

        return loss, first_layer_loss, predictions

    def init_vars(self):
        '''Sets all states and errors to zero vectors.'''
        for st_unit, err_unit in zip(self.st_units, self.err_units):
            st_unit.init_vars()
            err_unit.init_vars()
    
    def enable_layer_training(self, lyr):
        self.st_units[lyr].enable_training()
        self.err_units[lyr].enable_training()
    
    def disable_layer_training(self, lyr):
        self.st_units[lyr].disable_training()
        self.err_units[lyr].disable_training()


if __name__ == "__main__":
    state_unit = StateUnit(1, 5, 5)
    error_unit = ErrorUnit(1, 5, 5)

    print(len(list(state_unit.parameters())))

    v_params = list(state_unit.V.parameters())
    params = v_params + list(state_unit.LSTM_.parameters())
    print(len(params))

    print(v_params[0])
    print(v_params[1])

    print()

    print(state_unit.V.weight)
    print(state_unit.V.bias)

    print()

    print(state_unit.V.bias)
    print(state_unit.V.weight)

    # lstm = nn.LSTM(7, 4)
    # print(len(list(lstm.parameters())))
    # print(lstm.weight_ih_l0)

    print(len(list(error_unit.parameters())))
