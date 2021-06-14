import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # print(self.V.data)
        self.V.weight.data = torch.randn(self.V.weight.size())

        self.LSTM_ = nn.LSTMCell(
            input_size=thislayer_dim if is_top_layer else 2 * thislayer_dim,
            hidden_size=thislayer_dim)

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

        if self.layer_level == 1:
            self.recon = F.softmax(self.recon, dim=0)

    def set_state(self, input_char):
        self.state = input_char

    def init_vars(self):
        '''Sets state and reconstruction to zero, and set timestep to 0'''
        self.timestep = 0
        self.state = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        self.cell_state = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        # reconstructions at all other time points will be determined by the state
        self.recon = torch.randn(self.lowerlayer_dim, dtype=torch.float32)
        if self.layer_level == 1:
            self.recon = F.softmax(self.recon, dim=0)


class ErrorUnit(nn.Module):
    def __init__(self, layer_level, thislayer_dim, higherlayer_dim):
        super().__init__()
        self.layer_level = layer_level
        self.thislayer_dim = thislayer_dim
        self.higherlayer_dim = higherlayer_dim

        self.loss = 0

        self.init_vars()

        self.W = nn.Linear(thislayer_dim, higherlayer_dim)  # maps up to the next layer

    def forward(self, state, recon):
        self.timestep += 1
        if self.layer_level == 0:
            u = state * torch.log(recon) + (1 - state) * torch.log(1 - recon)

            # self.TD_err = -u 
            self.TD_err = state - recon

            self.loss = torch.sum(-u)

        else:
            self.TD_err = state - recon
        self.BU_err = self.W(self.TD_err.float())

    def init_vars(self):
        '''Sets TD_err and BU_err to zero, and set timestep to 0'''
        self.timestep = 0
        self.TD_err = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        # it shouldn't matter what we initialize this to; it will be determined by TD_err in all other iterations
        self.BU_err = torch.zeros(self.higherlayer_dim, dtype=torch.float32)


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


    def forward(self, input_sentence, iternumber):
        loss = 0
        first_layer_loss = 0
        lamda = 0.05
        if iternumber == 2990:
            pass
            stp = 0
        predictions = []
        for t in range(self.total_timesteps):
            # input_char at each t is a one-hot character encoding
            input_char = input_sentence[t]
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
                if lyr == 0:
                    loss += self.err_units[lyr].loss * lamda**lyr
                else:
                    loss += torch.sum(torch.abs(self.err_units[lyr].TD_err)) * lamda**lyr


            first_layer_loss += self.err_units[0].loss
            predictions.append(self.st_units[1].recon)
        
        return loss, first_layer_loss, predictions

    def init_vars(self):
        '''Sets all states and errors to zero vectors.'''
        for st_unit, err_unit in zip(self.st_units, self.err_units):
            st_unit.init_vars()
            err_unit.init_vars()


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
