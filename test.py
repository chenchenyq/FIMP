import torch
import cv2
import tqdm
import os
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
import dataset.dataset as myDataset
import dataset.Transforms as myTransforms
from models.change_classifier import ChangeClassifier as Model
import argparse


def evaluate(x1, x2, label, tool4metric):
    # All the tensors on the device:
    x1 = x1.to(device).float()
    x2 = x2.to(device).float()

    label = label.squeeze(1).to(device).long()

    # Evaluating the model:
    mask = model(x1, x2)

    # Feeding the comparison metric tool:
    mask = torch.argmax(mask, dim=1)
    tool4metric.update_cm(pr=mask.to("cpu").numpy(), gt=label.to("cpu").numpy())

    return mask.squeeze()


if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser(description="Parameter for data analysis, data cleaning and model training.")
    parser.add_argument("--dataname", default="WH", type=str, help="data name")
    parser.add_argument("--datapath", default="/home/t1/Chen/data/WH/data_256", type=str, help="data path")
    parser.add_argument("--modelpath", default="", type=str, help="model path")
    parser.add_argument("--vispath", default="/home/t1/Chen/TGRS/FIMP/vis", type=str, help="vis path")
    parser.add_argument('--test_batchsize', type=int, default=1, help='Val Batch size')
    parser.add_argument('--fuse_block', type=int, default=3, help='the number of UCPM')
    args = parser.parse_args()

    vispath = os.path.join(args.vispath, args.dataname)
    if not os.path.exists(vispath):
        os.mkdir(vispath)


    # Initialisation of the dataset
    data_path = args.datapath

    # Inizialitazion of dataset and dataloader:
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    test_Trans = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor()
    ])

    dataset = myDataset.Dataset("test", args.datapath, transform=test_Trans)
    test_loader = DataLoader(dataset, batch_size=args.test_batchsize)

    # Initialisation of the model and print model stat
    model = Model(num_classes=2, num=args.fuse_block)
    modelpath = args.modelpath
    model.load_state_dict(torch.load(modelpath))

    tool4metric = ConfuseMatrixMeter(n_class=2)

    # Set evaluation mode and cast the model to the desidered device
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    # loop to evaluate the model and print the metrics
    tool4metric.clear()
    with torch.no_grad():
        for img, label, name in tqdm.tqdm(test_loader):
            x1 = img[:, 0:3]
            x2 = img[:, 3:6]

            mask = evaluate(x1, x2, label, tool4metric)
            mask = mask*255

            cv2.imwrite(vispath + "/" + ''.join(name) + '.png', mask.cpu().numpy())

    scores_dictionary = tool4metric.get_scores()
    epoch_result = 'F1_score = {}, IoU = {}, Pre = {}, Recall = {}, Acc = {}'.format(
        scores_dictionary['F1_1'],
        scores_dictionary['iou_1'],
        scores_dictionary['precision_1'],
        scores_dictionary['recall_1'],
        scores_dictionary['acc'])
    print(epoch_result)
    print()










