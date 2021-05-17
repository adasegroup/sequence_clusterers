import torch
import torch.nn as nn

class CNN_Filters(nn.Module):
    def __init__(self, in_channels, number_of_filters, number_of_classes, steps, feature_space):
        super().__init__()
        #layer1
        for i in range(number_of_filters):
            setattr(self, 'layer_1_{}'.format(i), nn.Conv1d(in_channels = in_channels,\
                                                            out_channels = 16,\
                                                            kernel_size = 3,\
                                                            padding = 1))
        #layer2
        for i in range(number_of_filters):
            setattr(self, 'layer_2_{}'.format(i), nn.Conv1d(in_channels = 16,\
                                                            out_channels = 32,\
                                                            kernel_size = 3,\
                                                            padding = 1))
        #layer3
        for i in range(number_of_filters):
            setattr(self, 'layer_3_{}'.format(i), nn.Conv1d(in_channels = 32,\
                                                            out_channels = 64,\
                                                            kernel_size = 3,\
                                                            padding = 1))
            
        #linear
        for i in range(number_of_filters):
            setattr(self, 'linear_{}'.format(i), nn.Sequential(nn.Linear(64*steps[i], feature_space), nn.ReLU()))
            
        #predictor
        self.pred = nn.Linear(number_of_filters*feature_space, number_of_classes)

    def forward(self, X):
        #layer1
        out = [getattr(self, 'layer_1_{}'.format(i))(j) for i, j in enumerate(X)]
        #layer2
        out = [getattr(self, 'layer_2_{}'.format(i))(j) for i, j in enumerate(out)]
        #layer3
        out = [getattr(self, 'layer_3_{}'.format(i))(j).reshape(j.shape[0],-1) for i, j in enumerate(out)]
        #linear
        out = [getattr(self, 'linear_{}'.format(i))(j) for i, j in enumerate(out)]
        out = torch.cat(out, dim = 1)
        #predictor
        out = self.pred(out)
        return out
