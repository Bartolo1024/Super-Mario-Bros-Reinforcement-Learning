import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_shape, head_fets=((512,)), num_actions=4):
        super(DQN, self).__init__()
        in_channels, h, w = state_shape
        body = [self._conv_bn_block(in_ch, out_ch)
                for (in_ch, out_ch) in ((in_channels, 32), (32, 48), (48, 48), (48, 64), (64, 64))]
        h = int(h / 2 ** len(body))
        w = int(w / 2 ** len(body))
        self.body = nn.Sequential(*body)
        print([(64 * h * w, *head_fets)])
        head = [nn.Sequential(nn.Linear(in_fet, out_fet, num_actions), nn.ReLU())
                for in_fet, out_fet in [(64 * 8 * 8, *head_fets), (head_fets[-1], num_actions)]]
        self.head = nn.Sequential(*head)

    @staticmethod
    def _conv_bn_block(in_feats, out_feats):
        conv = nn.Conv2d(in_feats, out_feats, kernel_size=5, stride=2, padding=2)
        bn = nn.BatchNorm2d(out_feats)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x.view(x.size(0), -1))
        return x