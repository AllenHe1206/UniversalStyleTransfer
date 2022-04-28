import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size = 1)
        
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size =3)

    def forward(self,x):
        x = self.conv1(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = nn.ReLU(inplace=True)(self.conv2(x))
        return x
    
#######################################################

class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size = 1)

        self.conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size =3)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size =3)

    def forward(self,x):
        x = self.conv1(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        return x
    
#######################################################

class encoder3(nn.Module):
    def __init__(self):
        super(encoder3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size = 1)

        self.conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size =3)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size =3)
        
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size =3)
        
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size =3)

    def forward(self,x):
        x = self.conv1(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv6(x)
        x = nn.ReLU(inplace=True)(x)
        return x

#######################################################

class encoder4(nn.Module):
    def __init__(self):
        super(encoder4, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size = 1)

        self.conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size =3)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size =3)
        
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size =3)
        
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size =3)
        
        self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv9 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv10 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size =3)

    def forward(self,x):
        x = self.conv1(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv6(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv7(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv8(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv9(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv10(x)
        x = nn.ReLU(inplace=True)(x)
        return x

#######################################################

class encoder5(nn.Module):
    def __init__(self):
        super(encoder5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size = 1)

        self.conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size =3)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size =3)
        
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size =3)
        
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size =3)
        
        self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv9 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv10 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size =3)
        
        self.conv11 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv13 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv14 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)

    def forward(self,x):
        x = self.conv1(x)#!
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv6(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv7(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv8(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv9(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv10(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv11(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv12(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv13(x)
        x = nn.ReLU(inplace=True)(x)
        x,i = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv14(x)
        x = nn.ReLU(inplace=True)(x)
        return x
 
class decoder1(nn.Module):
    def __init__(self):
        super(decoder1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size =3)
        
    def forward(self,x):
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv1(x)
        return x

#######################################################

class decoder2(nn.Module):
    def __init__(self):
        super(decoder2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size =3)
        
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size =3)
        
    def forward(self,x):
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        return x

#######################################################

class decoder3(nn.Module):
    def __init__(self):
        super(decoder3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size =3)
        
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size =3)
        
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size =3)
        
    def forward(self,x):
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv5(x)
        return x
#######################################################

class decoder4(nn.Module):
    def __init__(self):
        super(decoder4, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size =3)
        
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size =3)
        
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size =3)
        
        self.conv7 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size =3)
        
        self.conv8 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv9 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size =3)
        
    def forward(self,x):
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv6(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv7(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv8(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv9(x)
        return x
    
#######################################################

class decoder5(nn.Module):
    def __init__(self):
        super(decoder5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size =3)
        
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size =3)
        
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size =3)
        
        self.conv9 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size =3)
        
        self.conv10 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size =3)
        
        self.conv11 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size =3)
        
        self.conv12 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size =3)
        
        self.conv13 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size =3)
        
    def forward(self,x):
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv1(x)##`
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv6(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv7(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv8(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv9(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv10(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv11(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.interpolate(x,scale_factor = 2)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv12(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.ReflectionPad2d((1,1,1,1))(x)
        x = self.conv13(x)
        return x   
