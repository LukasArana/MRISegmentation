import torch
from torch.utils.data import DataLoader, random_split
from dataset import MiceDataset 
from torch.nn import BCEWithLogitsLoss
from model import UNet, UNet_pretrained
from torch.optim import Adam
import sys
def training_loop(train_dl, model, loss_fn, optimizer):
    train_loss = 0
    for idx, data in enumerate(train_dl):
        img, seg = data
        img = img.cuda()
        seg = seg.cuda()
        output = model(img)	
        loss = loss_fn(output, seg)
	#Magic is done here :)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with open("/project/training_loss_every.txt", "a+") as f:
            f.write(f"{str(loss.item())} \n")
        train_loss += loss.item() / len(training_data)
    with open("/project/train_loss.txt", "a+") as f:
        f.write(f"{str(train_loss)} \n")

    return model, optimizer

K =33
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(K)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        num = int(sys.argv[1])
    else:
        num = 0
    if num == 0:
        model = UNet(1).cuda()
    elif num == 1:
        model = UNet_pretrained().cuda()
    print(model) 
    dataset = MiceDataset("/data/manifest-1686081801328/new_metadata.csv","/data/manifest-1686081801328")
    train_le = int(len(dataset) * 0.7)
    val_le = int(len(dataset) * 0.2)

    training_data, val_data, test_data = random_split(dataset, [train_le, val_le,len(dataset) - train_le - val_le ])

    train_dl = DataLoader(training_data, batch_size=16, num_workers = 4)
    val_dl = DataLoader(val_data, batch_size=16, num_workers = 4)
    test_dl = DataLoader(training_data, batch_size=16, num_workers = 4)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    
    for i in range(EPOCHS):
        print(f"epochs: {i}")
        model, optimizer = training_loop(train_dl, model, loss_fn, optimizer)
        with torch.no_grad():
        	val_loss = 0
	        for idx, data in enumerate(val_dl):
		        img, seg = data
		        img, seg = img.cuda(), seg.cuda()
		        output = model(img)
		        loss = loss_fn(output, seg)
		        val_loss += loss.item() / len(val_data)
	        with open("/project/val_loss.txt", "a+") as f:
		        f.write(f"{str(val_loss)} \n")
    torch.save(model.state_dict(), "/project/model_1.pth") 
