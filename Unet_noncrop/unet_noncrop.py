import torch
import torch.nn as nn
import torch.nn.functional as F


class Contracting(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(64, 64, 3, 1, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(128, 128, 3, 1, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU())    
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU())                          
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(1024, 1024, 3, 1, padding=1), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x1 = x
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x2 = x       
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x3 = x
        x = F.max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x4 = x
        x = F.max_pool2d(x, 2, 2)

        x = self.conv5(x)

        return x, x1, x2, x3, x4


class Expanding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_up1 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU())
        self.conv_up2 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU())
        self.conv_up3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(128, 128, 3, 1, padding=1), nn.ReLU())    
        self.conv_up4 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(64, 64, 3, 1, padding=1), nn.ReLU(), 
                                    nn.Conv2d(64, 1, 1, 1), nn.ReLU())
        self.upsample1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upsample2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upsample3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upsample4 = nn.ConvTranspose2d(128, 64, 2, 2)


    def forward(self, x, x1, x2, x3 ,x4):
        x = self.upsample1(x)
        x = torch.cat((x4, x), 1)
        x = self.conv_up1(x)

        x = self.upsample2(x)
        x = torch.cat((x3, x), 1)
        x = self.conv_up2(x)

        x = self.upsample3(x)
        x = torch.cat((x2, x), 1)
        x = self.conv_up3(x)

        x = self.upsample4(x)
        x = torch.cat((x1, x), 1)
        x = self.conv_up4(x)

        return x

class UNet_noncrop(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = Contracting()
        self.up = Expanding()

    def forward(self, x):
        x, x1, x2, x3, x4 = self.down(x)
        x = self.up(x, x1, x2, x3 ,x4)
        return x

if __name__ == '__main__':
    input = torch.randn(1,1,512,512)
    model = UNet_noncrop()
    output = model(input)
    print(output.size())
