
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class VAE(nn.Module):
    def __init__(self, latent_space, isTraining):
        super(VAE, self).__init__()
        self.z_dim = latent_space
        self.isTraining = isTraining

        # Define layer of encoder with leaky relu
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn0 = nn.BatchNorm2d(16) # default eps=1e-05

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.lRelu = nn.LeakyReLU(negative_slope= 0.2)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(8192, self.z_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.z_dim)

        self.fc21 = nn.Linear(self.z_dim, self.z_dim)
        self.fc22 = nn.Linear(self.z_dim, self.z_dim)

        # Sampling vector
        self.fc3 = nn.Linear(self.z_dim, self.z_dim)
        self.fc_bn3 = nn.BatchNorm1d(self.z_dim)
        self.fc4 = nn.Linear(self.z_dim, 8192)
        self.fc_bn4 = nn.BatchNorm1d(8192)
      
        # Decoder
        
        self.dconv0 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=4, stride=2, padding=1)
        self.bndc0 = nn.BatchNorm2d(64)

        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32,kernel_size=4, stride=2, padding=1)
        self.bndc1 = nn.BatchNorm2d(32)

        self.dconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16,kernel_size=4, stride=2, padding=1)
        self.bndc2 = nn.BatchNorm2d(16)

        self.dconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=3,kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig  = nn.Sigmoid()

    def reparametize(self, mu, logvar, isTraining):
        if isTraining:
        
            std = logvar.mul(0.5).exp_() 
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            return z

        else:
            return mu

    def encode(self, input):
        # Encoder
        # print(input.shape)
        conv0 = self.lRelu(self.bn0(self.conv0(input)))
        conv1 = self.lRelu(self.bn1(self.conv1(conv0)))
        conv2 = self.lRelu(self.bn2(self.conv2(conv1)))
        conv3 = self.lRelu(self.bn3(self.conv3(conv2)))

        # flatten ouput
        size = conv3.size(1) * conv3.size(2) * conv3.size(3)
        flatten = conv3.view(conv3.size(0), -1)

        # Latent vectors
        fc1 = self.relu(self.fc_bn1(self.fc1(flatten)))
        mu = self.fc21(fc1)
        logvar = self.fc22(fc1)

        return mu, logvar
        # return fc1


    def decode(self, z):
        # Decode
        # print(z.shape)
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 128, 8, 8)

        conv5 = self.relu(self.bndc0(self.dconv0(fc4)))
        conv6 = self.relu(self.bndc1(self.dconv1(conv5)))
        conv7 = self.relu(self.bndc2(self.dconv2(conv6)))
        out = self.tanh(self.dconv3(conv7).view(-1, 3, 128, 128))

        return out 

    def forward(self, input):
        # encoder
        mu, logvar = self.encode(input)
        
        # Reparameterize
        z = self.reparametize(mu, logvar, self.isTraining)
        
        # Decode
        out = self.decode(z)

        return out, mu, logvar


# class VAE(nn.Module):
#     def __init__(self, zsize, layer_count=3, channels=3):
#         super(VAE, self).__init__()

#         d = 128 #128
#         self.d = d
#         self.zsize = zsize

#         self.layer_count = layer_count

#         mul = 1
#         inputs = channels
#         for i in range(self.layer_count):
#             setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
#             setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
#             inputs = d * mul
#             mul *= 2

#         self.d_max = inputs

#         self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
#         self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

#         self.d1 = nn.Linear(zsize, inputs * 4 * 4)

#         mul = inputs // d // 2

#         for i in range(1, self.layer_count):
#             setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
#             setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
#             inputs = d * mul
#             mul //= 2

#         setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

#     def encode(self, x):
#         for i in range(self.layer_count):
#             x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

#         x = x.view(x.shape[0], self.d_max * 4 * 4)
#         h1 = self.fc1(x)
#         h2 = self.fc2(x)
#         return h1, h2

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def decode(self, x):
#         x = x.view(x.shape[0], self.zsize)
#         x = self.d1(x)
#         x = x.view(x.shape[0], self.d_max, 4, 4)
#         #x = self.deconv1_bn(x)
#         x = F.leaky_relu(x, 0.2)

#         for i in range(1, self.layer_count):
#             x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

#         x = torch.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         mu = mu.squeeze()
#         logvar = logvar.squeeze()
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)


# def normal_init(m, mean, std):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(mean, std)
#         m.bias.data.zero_()

# device = torch.device('cuda')
# model = VAE(100, 5, 3).to(device)
# model = summary(model, (3,128,128))