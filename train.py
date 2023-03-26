# created based on https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ray import tune
from ray.air import session

from dataset import PennFudanDataset
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torch

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
# set training config here
config = {
    'batch_size' : 2,
    'shuffle' : True,
    'lr' : 0.005,
    'momentum' : 0.9,
    'weight_decay' : 0.0005,
    'step_size' : 3,
    'gamma' : 0.1,
    'num_epochs' : 10,
    'raytune' : True
}

def train(config):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    # download dataset: 
    data_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'PennFudanPed')
    dataset = PennFudanDataset(data_root, get_transform(train=True))
    dataset_test = PennFudanDataset(data_root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config['lr'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config['step_size'],
                                                   gamma=config['gamma'])

    # let's train it for 10 epochs
    num_epochs = config['num_epochs']

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_test, device=device)
        bbox_metrics = evaluator.coco_eval['bbox'].stats
        segm_metrics = evaluator.coco_eval['segm'].stats
        if config['raytune']:
            session.report({"mAP": bbox_metrics[0]})

    print("Done")

if __name__ == "__main__":
    train(config)