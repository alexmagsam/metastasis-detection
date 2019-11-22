import torch
import torch.nn as nn
import torchvision.models as models


class CamelyonClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.pool = nn.MaxPool2d(3, 1)
        self.fc = nn.Sequential(nn.Linear(1280, 1), nn.Sigmoid())

        n_params = sum([p.numel() for p in self.parameters()])

        print("\n")
        print("# " * 50)
        print("MobileNet v2 initialized with {:.3e} parameters".format(n_params))
        print("# " * 50)
        print("\n")

    def forward(self, x):

        return self.fc(self.pool(self.backbone(x)).view(x.shape[0], -1))

    def print_modules(self):
        for idx, param in enumerate(self.modules()):
            print("Module : ", idx)
            print(param)
            print("\n")


if __name__ == '__main__':

    zeros = torch.zeros((2, 3, 96, 96))
    model = CamelyonClassifier()
    print(model(zeros).shape)
