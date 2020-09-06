# Variational autoencoder based on :
# https://github.com/WojciechMormul/vae

import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils

from tensorboardX import SummaryWriter

from vae import *

path = '/media/liz/TOSHIBA EXT/Data/LFW/lfw128'
path_result = '/media/liz/TOSHIBA EXT/Test/vae'
path_writer = ''
print_iter = 5

# configuration
latent_space    = 512
batch_size      = 4
learning_rate   = 0.0005
trai_iters      = 40

device = torch.device('cuda')

# defined loss

loss_bce = nn.BCELoss(reduction='mean')
loss_mse = nn.MSELoss(reduction='mean')

def train(model, optimizer, epoch, dataset):

    model.train()

    train_loss = 0
    kld_loss = 0
    bce_loss = 0

    for batch_idx, (data, _) in enumerate(dataset):
        
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, kld, bce = loss_function(recon_batch, data, mu, logvar)
        
        # recon_batch = model(data)
        # loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        kld_loss += kld.item()
        bce_loss += bce.item()

        optimizer.step()

        # if epoch % print_iter == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(dataset),
        #         100. * batch_idx / len(dataset),
        #         loss.item() / len(data)))

    return train_loss/len(dataset), kld_loss/len(dataset), bce_loss/len(dataset)
        
def test(model, epoch, dataset):

    model.eval()
    test_loss = 0
    kld_loss = 0
    bce_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(dataset):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, kld, bce = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            kld_loss += kld.item()
            bce_loss += bce.item()

            # test_loss += loss_function(recon_batch, data).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 3, 128, 128)[:n]])
                vutils.save_image(comparison.cpu(),path_result + '/reconstruction_' + str(epoch) + '.png', normalize=True)

    return test_loss/len(dataset), kld_loss/len(dataset), bce_loss/len(dataset)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.zeros_(m.bias.data)


def loss_function(recon_x, x, mu, logvar):
    # BCE = loss_bce(recon_x, x)
    MSE = loss_mse(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

    return 0.01*KLD + MSE, KLD, MSE

# def loss_function_(recon_x, x):
#     # BCE = loss_bce(recon_x, x)
#     MSE = loss_mse(recon_x, x)
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

#     # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

#     return MSE

def getDataset(path):
    data = datasets.ImageFolder(path, 
                    transform=transforms.Compose(
                                [transforms.Resize(128),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
         ]))

    NUM_CLASSES = len(data.classes)

    print("Size dataset: ", len(data))
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    print("Train dataset: ", train_size)
    print("Test dataset: ", test_size)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader

def main():
    # Define model
    model = VAE(latent_space, 5, 3).to(device)

    # xavier initialization 
    model.apply(weights_init)

    #optimiser Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss

    # Datasource
    train_dataset, test_dataset = getDataset(path)

    # summary writer
    writer_train = SummaryWriter('vae_5layer')
    writer_test = SummaryWriter('vae_5layer')


    # Train and test 
    for i in range(trai_iters):

        loss_train, kld_train, bce_train = train(model, optimizer, i, train_dataset)

        loss_test , kld_test, bce_test = test(model, i, test_dataset)

        writer_train.add_scalar("Loss", loss_train, i)
        # writer.add_scalar("KLD_train", loss_train, i)
        # writer.add_scalar("BCE_train", loss_train, i)
        writer_test.add_scalar("Loss", loss_test, i)
        # writer.add_scalar("KLD_test", loss_train, i)
        # writer.add_scalar("BCE_test", loss_train, i)

        print("%d Train: %.3f KLD: %.3f MSE: %.3f "%(i, loss_train, kld_train, bce_train))
        print("%d Test: %.3f KLD: %.3f MSE: %.3f "%(i, loss_test , kld_test, bce_test))

        with torch.no_grad():
            num_batch = 6
            sample = torch.randn(num_batch, latent_space).to(device)
            sample = model.decode(sample).cpu()
            vutils.save_image(sample.view(num_batch, 3, 128, 128),
                       path_result + 'sample_' + str(trai_iters) + '.png', normalize=True)


if __name__ == "__main__":
    main()


