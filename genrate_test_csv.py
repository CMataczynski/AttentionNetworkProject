import torch
from torch.utils.data import DataLoader
from utils import USGDataset
from model import ResidualAttentionModel_56, ResidualAttentionModel_92
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

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

    net = ResidualAttentionModel_92()
    net.load_state_dict(torch.load("./Res_92.pth"))
    net.to(device)
    idents = []
    predict = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):
            inputs = data['image'].to(device)
            ids = data['identifier']
            bs, ncrops, c, h, w = inputs.size()
            result = net(inputs.view(-1, c, h, w))
            result_avg = result.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(result_avg.data, 1)
            idents += ids.tolist()
            predict += predicted.tolist()

    out_data = {"id": idents, "label": predict}
    out_data = pd.DataFrame(out_data)
    out_data = out_data.sort_values(by="id")
    out_data.to_csv("submission92.csv", index=False)
    

