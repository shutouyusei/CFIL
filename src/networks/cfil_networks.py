import torch
import torch.nn as nn

#NOTE:input -> (C,H,W)
class StateEncoder(nn.Module):
    def __init__(self,input_size,representation_size,representation_final_size,H=224,W=256):
        super(StateEncoder,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, representation_size, kernel_size=5, stride=2, padding=2), # H/2, W/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/4, W/4
            
            nn.Conv2d(representation_size, representation_size*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/8, W/8

            nn.Conv2d(representation_size*2, representation_size*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # H/16, W/16
        )
        final_conv_output_channels = int(W/2)
        final_height = int(H/16)
        final_width = int(W/16)

        self._flattened_feature_dim = final_conv_output_channels * final_height * final_width

        self.fc = nn.Linear(self._flattened_feature_dim,representation_final_size)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ActionEncoder (nn.Module):
    def __init__(self,input_size,representation_size,output_size):
        super(ActionEncoder,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class StateTransitionModel(nn.Module):
    def __init__(self,input_size,representation_size,output_size):
        super(StateTransitionModel,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
