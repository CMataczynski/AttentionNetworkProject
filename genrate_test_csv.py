import torch
from torch.utils.data import DataLoader
from utils import USGDataset
from model import ResidualAttentionModel_56, ResidualAttentionModel_92
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU")

    testing_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    test_dataset = USGDataset("test_dataset.pkl", testing_transforms)

    test_dataloader = DataLoader(test_dataset, 16, shuffle=False, num_workers=0)

    net = ResidualAttentionModel_56()
    net.load_state_dict(torch.load("./Res_56.pth"))
    net.to(device)
    idents = []
    predict = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs = data['image'].to(device)
            ids = data['id'].to(device)
            bs, ncrops, c, h, w = inputs.size()
            result = net(inputs.view(-1, c, h, w))
            result_avg = result.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(result_avg.data, 1)
            idents += ids.tolist()
            predict += predicted.tolist()

    import csv

    with open('submission.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['id'] + ['label'])
        for i in range(len(idents)):
            spamwriter.writerow([idents[i], predict[i]])
    

