# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
import glob
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as image
from tqdm import tqdm

img_path = 'dataset/'

transform_train = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(), ])

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

classes = ('NG', 'OK')

batchsize = 8
numworkers = 2

import os
num_classes = 2
number_per_class = {}

for i in range(num_classes):
    number_per_class[i] = 0

# def custom_imsave(img, label):
#     path = 'dataset/' + str(label) + '/'
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     img = img.numpy()
#     img = np.transpose(img, (1, 2, 0))
#     image.imsave(path + str(number_per_class[label]) + '.bmp', img)
#     number_per_class[label] += 1



# def process():
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         print("[ Current Batch Index: " + str(batch_idx) + " ]")
#         for i in range(inputs.size(0)):
#             custom_imsave(inputs[i], targets[i].item())
# process()

# print("Done saving Image")
from PIL import Image
from matplotlib.pyplot import imshow

img = Image.open('dataset/Train/1/0.bmp')
imshow(np.asarray(img))

from torchvision.datasets import ImageFolder


train_dataset = ImageFolder(root='./dataset/Train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=numworkers)

classes = ('NG', 'OK')

def custom_imshow(img):
#    img = img.numpy().reshape((1, 2448, 2058))
    # plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
    plt.show()

# def process():
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to('cuda'), targets.to('cuda')
#         # custom_imshow(inputs)
#         print(inputs.shape)

# process()

dataiter = iter(train_loader)
image, labels = dataiter.next()

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(511*609*3, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#        print(x.shape)
        x = x.view(-1,511*609*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#net = Net()
net = Net().cuda()
print('Finished forward')

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

print('Finished optimizer')
for epoch in range(30):

    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader, 0)):
 #       inputs = inputs
 #       labels = labels

        optimizer.zero_grad()
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 #       optimizer.zero_grad()

        running_loss += loss.item()
        if i%10 ==9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/10 ))
            running_loss = 0.0
print('Finished Training')

PATH = './wi_DispInsp_net.pth'
torch.save(net.state_dict(), PATH)

test_dataset = ImageFolder(root='./dataset/Test', transform=transform_train)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=numworkers)

print('Finshed test loader')

dataiter = iter(test_loader)
images, labels = dataiter.next()

images = images.cuda()
labels = labels.cuda()

print('Finished dataiter')

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batchsize)))


net = Net().cuda()
#net = Net()
net.load_state_dict(torch.load(PATH))
print('Finished load trained model')

outputs= net(images)
print('Finished test net')
_, predicted = torch.max(outputs, 1)
print('Finished max')
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batchsize)))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Finished whole test')
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in test_loader:
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
print('Finished whole test print')

for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i]/class_total[i]))