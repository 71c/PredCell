import torch
import torch.nn as nn


class StateUnit(nn.Module):
    def __init__(self, layer_level, thislayer_dim, lowerlayer_dim, is_top_layer=False):
        super().__init__()
        
        self.layer_level = layer_level
        self.timestep = 0
        self.is_top_layer = is_top_layer        
        self.thislayer_dim = thislayer_dim
        self.lowerlayer_dim = lowerlayer_dim

        self.init_vars()

        # maps from this layer to the lower layer
        # Note: includes a bias
        self.V = nn.Linear(thislayer_dim, lowerlayer_dim)
        self.LSTM_ = nn.LSTM(
            input_size=thislayer_dim if is_top_layer else 2 * thislayer_dim,
            hidden_size=thislayer_dim, num_layers=1)

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
    
    def init_vars(self):
        '''Sets state and reconstruction to zero'''
        self.state = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        # reconstructions at all other time points will be determined by the state
        self.recon = torch.zeros(self.lowerlayer_dim, dtype=torch.float32)


class ErrorUnit(nn.Module):
    def __init__(self, layer_level, thislayer_dim, higherlayer_dim):
        super().__init__()
        self.layer_level = layer_level
        self.timestep = 0
        self.thislayer_dim = thislayer_dim
        self.higherlayer_dim = higherlayer_dim

        self.init_vars()
        self.W = nn.Linear(thislayer_dim, higherlayer_dim)  # maps up to the next layer

    def forward(self, state_, recon_):
        self.timestep += 1
        #self.TD_err = torch.abs(state_ - recon_)
        self.TD_err = state_ - recon_
        self.BU_err = self.W(self.TD_err.float())
    
    def init_vars(self):
        '''Sets TD_err and BU_err to zero'''
        self.TD_err = torch.zeros(self.thislayer_dim, dtype=torch.float32)
        # it shouldn't matter what we initialize this to; it will be determined by TD_err in all other iterations
        self.BU_err = torch.zeros(self.higherlayer_dim, dtype=torch.float32)


class PredCell(nn.Module):
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
                self.st_units.append(StateUnit(lyr, self.numchars, self.numchars))
                self.err_units.append(ErrorUnit(lyr, self.numchars, hidden_dim))
            elif lyr < self.num_layers - 1 and lyr > 0:
                if lyr == 1:
                    self.st_units.append(StateUnit(lyr, hidden_dim, self.numchars))
                else:
                    self.st_units.append(StateUnit(lyr, hidden_dim, hidden_dim))
                self.err_units.append(ErrorUnit(lyr, hidden_dim, hidden_dim))
            else:
                self.st_units.append(StateUnit(lyr, hidden_dim, hidden_dim, is_top_layer=True))
                self.err_units.append(ErrorUnit(lyr, hidden_dim, hidden_dim))

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

    def init_vars(self):
        '''Sets all states and errors to zero vectors.'''
        for st_unit, err_unit in zip(self.st_units, self.err_units):
            st_unit.init_vars()
            err_unit.init_vars()


if __name__ == "__main__":
    predcell = PredCell(3, 100, 128)
    predcell.init_vars()
