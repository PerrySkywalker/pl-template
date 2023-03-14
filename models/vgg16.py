import torch.nn as nn
import torch

class vgg16(nn.Module):
    def __init__(self, in_channels=3, n_classses=2):
        super(vgg16, self).__init__()
        self.in_channels = in_channels
        self.out_dim = n_classses
        #list 
        #1*3*256*256
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            #256*256 128*128
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            #128*128
            #64*64
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            #64*64
            #32*32
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            #32*32
            #512*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            #1*512*8*8
        )
        self.classifier = nn.Sequential(
            #1*32768
            #(1*32768) * (32768 *4096) = 1 * 4096
            nn.Flatten(),
            nn.Linear(in_features=8*8*512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            #(1*4096) * (4096 *4096) = 1 * 4096
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            #(1*4096) * (4096 * 2) = 1 * 2
            #0->cat 
            #1->dog
            #3*256*256 -> 1*2  [0.7,0.8] argmax() = 1
            nn.Linear(in_features=4096, out_features=self.out_dim)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.classifier(x) #(0.7, 0.8)
        return logits

if __name__ == "__main__":
    #256 = 2^
    #3*3 5*5 7*7 stride padding 3//2 5//2 7//2
    x = torch.rand(size=(8, 3, 256, 256))
    vgg = vgg16(in_channels=3, n_classses=2)
    y = vgg(x)
    print(y.size())


