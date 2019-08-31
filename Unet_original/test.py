from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import utils
import train
import numpy as np


def test(args, model, device, test_loader):
    model.eval()

    model.load_state_dict(torch.load(args.checkpoint_dir + "unet_train.pth")['model_state_dict'])
    
    test_loss = 0
    correct = 0
    writer = SummaryWriter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            pred = model(data)
            pred = torch.sigmoid(pred)
            target_crop = crop_tensor(target, pred.size()[2])

            criterion = nn.BCELoss()

            pred_flat = pred.reshape(-1)
            target_flat = target_crop.reshape(-1)
            test_loss += criterion(pred_flat, target_flat).item()
            
            pred_bin = (pred > 0.5).float()
            correct += (pred_bin.eq(target_crop.view_as(pred_bin)).sum().item() / (np.size(target_crop.cpu().detach().numpy()))) 

            
        # torch.save(pred, 'pred_tensor.pt')
        # torch.save(target_crop, 'target_tensor.pt')
        # im_show([target_crop, pred],16)
    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))

def get_args():
      
    parser = argparse.ArgumentParser(description='PyTorch U-Net original')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
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
    parser.add_argument('--checkpoint_dir', default='/home/raymondlab/data/Tracy/DDD/checkpoints/',
                        help='Checkpoint saving directory')                    
    parser.add_argument('--tb_dir', dest='tensorboard', default='/home/raymondlab/data/Tracy/DDD/',
                        help='Tensorboard saving directory')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    writer = SummaryWriter()

    # load the training data
    test_path = '/dataset/train/image/'
    test_target_path = '/dataset/train/label/'
    load_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = LocalDataset(test_path, test_target_path, transform=load_transform)
    test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = UNet().to(device)
    test(args, model, device, test_loader)

if __name__ == '__main__':
    main()

    # python3 test.py 