import os.path
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

kwargs = {}
train_data = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=100, shuffle=True, **kwargs)
test_data = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=100, shuffle=True, **kwargs)


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 80, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(1280, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, 1)


if os.path.isfile("net.pt"):
    model = torch.load("net.pt")
    print("yes")
else:
    model = net()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = f.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        out = model(data)
        loss += f.nll_loss(out, target)
        prediction = out.argmax(1, keepdim=True)
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    loss = loss/len(test_data.dataset)
    print(f"Average_loss: {loss}")
    print(f"Accuracy: {100.0*correct/len(test_data.dataset)}")


for epoch in range(1, 50):
    print(epoch)
    train(epoch)
    torch.save(model, "net.pt")
    test()