import torch
from torch import nn


class PlotQA_Regressor_v20(nn.Module):
    def __init__(self, config):
        super(PlotQA_Regressor_v20, self).__init__()
        self.txt_pipe = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(config.hidden_size, 512),
                                      nn.LeakyReLU(),
                                      nn.Linear(512, 256),
                                      nn.LeakyReLU(),
                                      nn.Linear(256, 256)
                                      )

        self.vis_pipe = nn.Sequential(nn.Linear(config.v_hidden_size, config.v_hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(config.v_hidden_size, 512),
                                      nn.LeakyReLU(),
                                      nn.Linear(512, 256),
                                      nn.LeakyReLU(),
                                      nn.Linear(256, 256)
                                      )

        self.fusion = nn.Sequential(nn.Linear(512, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 1),
                                    nn.Tanh()
                                    )

    def forward(self, hv_0, hw_0):
        hw = self.txt_pipe(hw_0)
        hv = self.vis_pipe(hv_0)

        prefusion = torch.cat((hv, hw), axis=-1)
        fusion = self.fusion(prefusion)
        return fusion.squeeze(-1)


class DVQA_Regressor_v20_CE(nn.Module):
    def __init__(self, config):
        super(DVQA_Regressor_v20_CE, self).__init__()
        self.txt_pipe = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(config.hidden_size, 512),
                                      nn.LeakyReLU(),
                                      nn.Linear(512, 256),
                                      nn.LeakyReLU(),
                                      nn.Linear(256, 256)
                                      )

        self.vis_pipe = nn.Sequential(nn.Linear(config.v_hidden_size, config.v_hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(config.v_hidden_size, 512),
                                      nn.LeakyReLU(),
                                      nn.Linear(512, 256),
                                      nn.LeakyReLU(),
                                      nn.Linear(256, 256)
                                      )

        self.ce_fusion = nn.Sequential(nn.Linear(512, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 65),
                                    nn.Softmax()
                                    )

    def forward(self, hv_0, hw_0):
        hw = self.txt_pipe(hw_0)
        hv = self.vis_pipe(hv_0)

        prefusion = torch.cat((hv, hw), axis=-1)
        fusion = self.ce_fusion(prefusion)
        return fusion.squeeze(-1)
