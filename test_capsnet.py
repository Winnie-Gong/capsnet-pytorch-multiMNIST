import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm
import random

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
N_EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MODEL_PATH = "model/model.pt"

'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'smallnorb':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28


def train(model, optimizer, train_loader, epoch, multi=False):
    capsule_net = model
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        b_size = BATCH_SIZE

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        if multi:
            b_size = b_size/2
            for i in range(50):
                j = random.randint(50,99)
                while np.argmax(target[j]) == np.argmax(target[i]):
                    j = random.randint(50,99)
                data[i] = torch.add(data[i], data[j])
                target[i] = torch.add(target[i], target[j])
            data = data[0:50,:,:,:]
            target = target[0:50,:]

        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()
        if multi:
            correct = 0
            for i in range(50):
                if (masked[i].data.cpu().numpy() == target[i].data.cpu().numpy()).all():
                    correct += 1
        else:
            correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))

        reconstruc_error = torch.abs(torch.sum(torch.sub(reconstructions,data)))
        train_loss = loss.item()
        total_loss += train_loss
        if batch_id % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, reconstruc error: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(b_size),
                reconstruc_error / float(b_size),
                train_loss / float(b_size)
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch,N_EPOCHS,total_loss / len(train_loader.dataset)))


def test(capsule_net, test_loader, epoch, multi=False):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        b_size = len(test_loader.dataset)

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)

        if multi:
            b_size = b_size/2
            for i in range(50):
                j = random.randint(50,99)
                while np.argmax(target[j]) == np.argmax(target[i]):
                    j = random.randint(50,99)
                data[i] = torch.add(data[i], data[j])
                target[i] = torch.add(target[i], target[j])
            data = data[0:50,:,:,:]
            target = target[0:50,:]

        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        if multi:
            for i in range(50):
                if (masked[i].data.cpu().numpy() == target[i].data.cpu().numpy()).all():
                    correct += 1
        else:
            correct += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        reconstruc_error = torch.abs(torch.sum(torch.sub(reconstructions,data)))

        routing_in = capsule_net.routing_in.cpu().detach().numpy()
        routing_out = capsule_net.routing_out.cpu().detach().numpy()
        for i in range(10):
            np.savetxt('routing_data/routing_in_'+str(i)+'.csv', routing_in[:, i, :], delimiter=',')
        np.savetxt('routing_data/routing_out.csv', routing_out, delimiter=',')

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, reconstruc error: {:.6}, loss: {:.6f}".format(epoch, N_EPOCHS, correct / b_size,
                                            reconstruc_error / b_size, test_loss / len(test_loader)))
    return correct / b_size, reconstruc_error / b_size


if __name__ == '__main__':
    torch.manual_seed(1)
    # dataset = 'cifar10'
    dataset = 'mnist'
    # dataset = 'smallnorb'
    config = Config(dataset)
    mnist = Dataset(dataset, BATCH_SIZE)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters())

    acc_track = np.zeros(N_EPOCHS)
    rec_acc_track = np.zeros(N_EPOCHS)
    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e, multi=False)
        torch.save(capsule_net,"model/3_iter/model_"+str(e)+".pt")
        acc, rec_acc = test(capsule_net, mnist.test_loader, e, multi=False)
        acc_track[e-1] = acc
        rec_acc_track[e-1] = rec_acc
    np.savetxt('test_acc/3_iter_acc.csv', acc_track, delimiter=',')
    np.savetxt('test_acc/3_iter_rec_acc.csv', rec_acc_track, delimiter=',')

    acc_track = np.zeros(N_EPOCHS)
    rec_acc_track = np.zeros(N_EPOCHS)
    capsule_net = CapsNet(config, iter_n=1)
    capsule_net = torch.nn.DataParallel(capsule_net)
    capsule_net = capsule_net.module
    optimizer = torch.optim.Adam(capsule_net.parameters())
    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e, multi=False)
        torch.save(capsule_net,"model/1_iter/model_"+str(e)+".pt")
        acc, rec_acc = test(capsule_net, mnist.test_loader, e, multi=False)
        acc_track[e-1] = acc
        rec_acc_track[e-1] = rec_acc
    np.savetxt('test_acc/1_iter_acc.csv', acc_track, delimiter=',')
    np.savetxt('test_acc/1_iter_rec_acc.csv', rec_acc_track, delimiter=',')

    acc_track = np.zeros(N_EPOCHS)
    rec_acc_track = np.zeros(N_EPOCHS)
    capsule_net = CapsNet(config, multi=True)
    capsule_net = torch.nn.DataParallel(capsule_net)
    capsule_net = capsule_net.module
    optimizer = torch.optim.Adam(capsule_net.parameters())
    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e, multi=True)
        torch.save(capsule_net,"model/multi_3_iter/model_"+str(e)+".pt")
        acc, rec_acc = test(capsule_net, mnist.test_loader, e, multi=True)
        acc_track[e-1] = acc
        rec_acc_track[e-1] = rec_acc
    np.savetxt('test_acc/m3_iter_acc.csv', acc_track, delimiter=',')
    np.savetxt('test_acc/m3_iter_rec_acc.csv', rec_acc_track, delimiter=',')

    acc_track = np.zeros(N_EPOCHS)
    rec_acc_track = np.zeros(N_EPOCHS)
    capsule_net = CapsNet(config, multi=True, iter_n=1)
    capsule_net = torch.nn.DataParallel(capsule_net)
    capsule_net = capsule_net.module
    optimizer = torch.optim.Adam(capsule_net.parameters())
    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e, multi=True)
        torch.save(capsule_net,"model/multi_1_iter/model_"+str(e)+".pt")
        acc, rec_acc = test(capsule_net, mnist.test_loader, e, multi=True)
        acc_track[e-1] = acc
        rec_acc_track[e-1] = rec_acc
    np.savetxt('test_acc/m1_iter_acc.csv', acc_track, delimiter=',')
    np.savetxt('test_acc/m1_iter_rec_acc.csv', rec_acc_track, delimiter=',')
    # model = torch.load("model/model_0.pt")
    # model.eval()
    # test(model, mnist.test_loader, 1)