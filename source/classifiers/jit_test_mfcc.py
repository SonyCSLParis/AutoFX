from torch import nn
import torch
import torchaudio


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.transform = torchaudio.transforms.MFCC()

    def forward(self, audio):
        return self.transform(audio)


test = Test()
print(torch.jit.script(test))
