from torch import nn
import torch
import torch.nn.functional as F
import ipdb
import torchsnooper
#from nowcasting.config import cfg
device = torch.device('cuda:0')

class ConvGRU(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv_xz = nn.Conv2d(input_channel+num_filter,num_filter,kernel_size,stride=1,padding=1)
        self.conv_xr = nn.Conv2d(input_channel+num_filter,num_filter,kernel_size,stride=1,padding=1)
        self.conv_xh = nn.Conv2d(input_channel+num_filter,num_filter,kernel_size,stride=1,padding=1)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.

        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    #@torchsnooper.snoop()
    def forward(self, inputs=None, states=None, seq_len=10):

        if states is None:
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(device)
        else:
            h,_ = states
        h_prev =h

        outputs = []
        for index in range(seq_len):
            #print(index)
            #ipdb.set_trace()
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(device)
            else:
                x = inputs[index, ...]
            #print(x.size())
            #print(h)
            cat_xz = torch.cat([x, h], dim=1)
            cat_xr = torch.cat([x, h], dim=1)
            conv_xz = self.conv_xz(cat_xz)
            conv_xr = self.conv_xz(cat_xr)

            zgate = torch.sigmoid(conv_xz)
            rgate = torch.sigmoid(conv_xr)
            combined = torch.cat((x,rgate*h_prev),1)
            ht = self.conv_xh(combined)
            ht = torch.tanh(ht)
            h_next = (1-zgate)*h_prev + zgate*ht


            outputs.append(h_next)
            h_prev = h_next
        return torch.stack(outputs), (h_next,None)

