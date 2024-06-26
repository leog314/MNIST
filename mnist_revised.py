import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

kwargs = {}
train_data = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=64, shuffle=True, **kwargs)
test_data = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=64, shuffle=True, **kwargs)

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv(x)
        x += shortcut
        return self.gelu(x)

class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()

        self.inp = nn.Sequential(
            nn.Conv2d(1, 64,5, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.res_block = nn.ModuleList([ResBlock() for _ in range(9)])
        # (batch, 64, 24, 24)
        self.fcb = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Flatten(1, 3),
            nn.Linear(576, 10),
            nn.LogSoftmax(1)
        )

        self.crit = nn.NLLLoss()
        self.optim = optim.RMSprop(self.parameters())

    def forward(self, x):
        c = self.inp(x)
        for Block in self.res_block:
            c = Block(c)
        return self.fcb(c)

model = mnist()

def training():
    model.train()
    for (data, target) in train_data:
        model.optim.zero_grad(True)

        out = model(data)
        loss = model.crit(out, target)
        loss.backward()
        model.optim.step()


def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        with torch.no_grad():
            out = model(data)

            loss += model.crit(out, target)
            prediction = out.argmax(1, keepdim=True)
            correct += prediction.eq(target.data.view_as(prediction)).sum()

    loss = loss/len(test_data.dataset)
    print(f"Average_loss: {loss}")
    print(f"Accuracy: {100.*correct/len(test_data.dataset)}")


for epoch in range(20):
    print(epoch)
    training()
    test()

torch.save(model, "mnist2.pt")