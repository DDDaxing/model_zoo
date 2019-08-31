
from unet_noncrop import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import *
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    writer = SummaryWriter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        pred = torch.sigmoid(pred)

        criterion = nn.BCELoss()
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        loss = criterion(pred_flat, target_flat)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # im_show([target_crop, pred],epoch)

    if epoch % 20 ==0:
        im_show([target, pred],epoch)

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))
    writer.add_scalar('Train_Loss', (epoch_loss/len(train_loader)), epoch)


def get_args():
    
    parser = argparse.ArgumentParser(description='PyTorch U-Net noncropped version')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                                        help='input batch size for testing (default: 5)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.99, metavar='M',
                                        help='SGD momentum (default: 0.99)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                                        help='how many batches to wait before logging training status')    
    parser.add_argument('--save-model', action='store_true', default=False,
                                        help='For Saving the current Model')
    parser.add_argument('--checkpoint_dir', default='/home/raymondlab/data/Tracy/DDD/Unet/DL/Unet_noncrop/checkpoints/',
                                        help='Checkpoint saving directory')                    
    parser.add_argument('--tb_dir', default='/home/raymondlab/data/Tracy/DDD/tb/',
                                        help='Tensorboard saving directory')
    args = parser.parse_args()

    return args


# def checkpoint(epoch, args):
#     checkpoint_path = args.checkpoint_dir + "model_epoch_{}.pth".format(epoch)
#     torch.save(model.state_dict(),checkpoint_path)


def main():
    
    args = get_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # writer = SummaryWriter(args.tb_dir)
    
    # load the training data
    train_path = '/home/raymondlab/data/Tracy/DDD/dataset/membrane/train/image/'
    train_target_path = '/home/raymondlab/data/Tracy/DDD/dataset/membrane/train/label/'
    load_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = LocalDataset(train_path, train_target_path,transform=load_transform)
    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # start training
    model = UNet_noncrop().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if (args.save_model):
            checkpoint_path = args.checkpoint_dir + "model_epoch_{}.pth".format(epoch)
            torch.save(model.state_dict(),checkpoint_path)

    if (args.save_model):
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, args.checkpoint_dir + "unet_noncrop_train.pt")
        

if __name__ == '__main__':
    main()

    # run with python3 train.py --save-model --epochs 300