import math
import statistics
import torch
import model_2
from sklearn.metrics import confusion_matrix
from args import args
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sever(object):
    def __init__(self, args, eva_dataset, global_model):

        if args.dataset in ['mnist', 'fmnist']:
            self.model = model_2.LeNet5(args.num_classes)
        elif args.dataset in ['cifar100']:
            self.model = model_2.VGG11(args.num_classes)
        elif args.dataset in ['cifar10']:
            self.model = model_2.Resnet18(args.num_classes)
        elif args.dataset in ['tinyimagenet', 'imagenet']:
            self.model = model_2.ViT_B16()
        self.eva_loader = torch.utils.data.DataLoader(eva_dataset, batch_size=64, shuffle=True)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(global_model)
        self.model.to(device)


    def evaluate(self, return_raw=False):
        # self.model.to(device)
        self.model.eval()
        correct = 0
        data_size = 0

        true_labels = []
        predicted_labels = []

        for batch_id, batch in enumerate(self.eva_loader):
            data, target = batch
            data_size += data.size()[0]
            data = data.to(device)
            target = target.to(device)

            output = self.model(data)
            pred = output.data.max(1)[1]

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        eval_acc = 100.0 * (float(correct) / float(data_size))

        if return_raw:
            return correct, data_size
        else:
            return eval_acc


