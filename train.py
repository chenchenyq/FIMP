import argparse
import os
import shutil
import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dataset.dataset as myDataset
import dataset.Transforms as myTransforms
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from tqdm import tqdm
import sys


def train(dataset_train, dataset_val, model, optimizer, scheduler, logpath, writer, epochs, device):

    model = model.to(device)
    tool4metric = ConfuseMatrixMeter(n_class=args.num_class)

    def training_phase(epc):
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        loop = tqdm(dataset_train, file=sys.stdout)
        for img, label, name in loop:
            optimizer.zero_grad()
            x1 = img[:, 0:3]
            x2 = img[:, 3:6]

            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            label = label.squeeze(1).to(device).long()


            # Evaluating the model:
            mask = model(x1, x2)

            bin_loss = nn.CrossEntropyLoss()(mask, label)
            loss = bin_loss

            # Reset the gradients:
            loss.backward()
            optimizer.step()

            # Track metrics:
            epoch_loss += loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        # scheduler step
        scheduler.step()

        epoch_loss /= len(dataset_train)

        #########
        print("Train Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        writer.flush()


    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        loop = tqdm(dataset_val, file=sys.stdout)
        with torch.no_grad():
            for img, label, name in loop:
                x1 = img[:, 0:3]
                x2 = img[:, 3:6]

                x1 = x1.to(device).float()
                x2 = x2.to(device).float()

                label = label.to(device).long()

                # Evaluating the model:
                mask = model(x1, x2)

                bin_loss = nn.CrossEntropyLoss()(mask, label.squeeze(1))
                loss = bin_loss

                epoch_loss_eval += loss
                mask = torch.argmax(mask, dim=1)
                tool4metric.update_cm(pr=mask.to("cpu").numpy(), gt=label.to("cpu").numpy())



        epoch_loss_eval /= len(dataset_val)
        print("Val Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()

        print('acc = {}, mIoU = {}, F1_1 = {},  iou_1= {}'
              .format(scores_dictionary['acc'], scores_dictionary['miou'],
                      scores_dictionary['F1_1'], scores_dictionary['iou_1']))

        return scores_dictionary['F1_1']

    bestscore = 0

    for epc in range(epochs):
        training_phase(epc)
        sc = validation_phase(epc)

        if sc > bestscore:
            bestscore = sc
            torch.save(model.state_dict(), os.path.join(logpath, "E{}_s{}.pth".format(epc, bestscore)))


def run(args):
    # set the random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.logpath)

    # Inizialitazion of dataset and dataloader:
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    train_Trans = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    val_Trans = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    traindata = myDataset.Dataset("train", args.datapath, transform=train_Trans)
    data_loader_train = DataLoader(traindata, batch_size=args.train_batchsize, num_workers=4, shuffle=True)
    validationdata = myDataset.Dataset("val", args.datapath, transform=val_Trans)
    data_loader_val = DataLoader(validationdata, batch_size=args.val_batchsize, num_workers=4, shuffle=True)

    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model(num_classes=args.num_class, num=args.fuse_block)

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    # choose the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-4)

    # copy the configurations
    _ = shutil.copytree("./models", os.path.join(args.logpath, "models"),)

    train(data_loader_train, data_loader_val, model, optimizer, scheduler, args.logpath, writer, epochs=args.epoch, device=device)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter for data analysis, data cleaning and model training.")
    parser.add_argument("--dataname", default="SYSU", type=str, help="data name")
    parser.add_argument("--datapath", default="/home/t1/Chen/data/SYSU-CD/data_256", type=str, help="data path")
    parser.add_argument("--logpath", default="", type=str, help="checkpoints log path")
    parser.add_argument('--num_class', type=int, default=2, help='the number of classes')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument("--seed", default=42, type=int, help="randm seeds")
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='max epoch')
    parser.add_argument('--train_batchsize', type=int, default=32, help='Train Batch size')
    parser.add_argument('--val_batchsize', type=int, default=32, help='Val Batch size')
    parser.add_argument('--fuse_block', type=int, default=3, help='the number of UCPM')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ''(only applicable to non-distributed training)')
    args = parser.parse_args()

    # create log dir if it doesn't exists
    modelpath = os.path.join(args.logpath, args.dataname)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(modelpath)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    args.logpath = os.path.join(modelpath, "run_%04d" % num_run + "/")

    run(args)
