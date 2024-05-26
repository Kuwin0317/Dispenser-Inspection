# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
import glob
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from tqdm import tqdm
from torchvision.models import resnet18
from PIL import Image
from matplotlib.pyplot import imshow


model = resnet18()
finalconv_name = 'layer4'

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (500, 500)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
#        print(weight_softmax.shape)
#        print(feature_conv.shape)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))

        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=3)

img_path = 'dataset/'

transform_train = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                      transforms.Resize(32),
#                                      transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

transform_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                      transforms.Resize(32),
#                                      transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])


classes = ('OK', 'NG', 'OF') #OK, NG, OVERFLOW

batchsize = 1
numworkers = 2
num_classes = 3
number_per_class = {}

for i in range(num_classes):
    number_per_class[i] = 0

#img = Image.open('dataset/Train/1/0.jpg')
#imshow(np.asarray(img))

from torchvision.datasets import ImageFolder


train_dataset = ImageFolder(root='./dataset/Train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=numworkers)
test_dataset = ImageFolder(root='./dataset/Test', transform=transform_train)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers)

dataiter = iter(train_loader)
image, labels = dataiter.next()


net = model.cuda()
print('Finished forward')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 60], gamma=0.1, last_epoch=-1, verbose=True)
print('Finished optimizer')

# for epoch in range(50):
#     net.train()
#     running_loss = 0.0
#     for i, (inputs, labels) in tqdm(enumerate(train_loader, 0)):
#
#         optimizer.zero_grad()
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#
#         outputs = net(inputs)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i%100 ==99:
#             print('[%d, %5d] loss: %.3f lr: %.4f' % (epoch + 1, i + 1, running_loss/10 , optimizer.param_groups[0]['lr'] ))
#             running_loss = 0.0
#
#     scheduler.step(epoch)
#     correct = 0
#     total = 0
#
#     net.eval()
#     class_correct = list(0. for i in range(num_classes))
#     class_total = list(0. for i in range(num_classes))
#     class_correct_rate = list(0. for i in range(num_classes))
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#
#             images = images.cuda()
#             labels = labels.cuda()
#
#             outputs = net(images)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(3):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#
#     for i in range(num_classes):
#         class_correct_rate[i] = 100 * class_correct[i] / class_total[i]
#         print('Accuracy of %2s : %2d %%' % (
#             classes[i], 100 * class_correct[i] / class_total[i]))
#
#
#     if class_correct_rate[0] == 100 and class_correct_rate[1] == 100 and class_correct_rate[2] == 100:
#         PATH = './wi_DispInsp_net_epoch%d.pth'%epoch
#         torch.save(net.state_dict(), PATH)
#     else:
#         print('We can not save dictionary, because accuracy is not 100% ')
#
PATH = './wi_DispInsp_net_epoch49.pth'
# torch.save(net.state_dict(), PATH)
# print('Finished Training')

dataiter = iter(test_loader)
images, labels = dataiter.next()

images = images.cuda()
labels = labels.cuda()

print('Finished dataiter')

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batchsize)))


net = model.cuda()
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
net.eval()
with torch.no_grad():
#    print(os.listdir())
    for data in test_loader:
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        probs, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # if(predicted != labels):
        #     print()


print('Finished whole test')
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(model.parameters())

tensor_to_image = transforms.ToPILImage()
num = 0
with torch.no_grad():
    for data in test_loader:
        num += 1
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        probs, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()


        # for i in range(num_classes):
        #     label = labels[i]
        #     class_correct[label] += c[i].item()
        #     class_total[label] += 1

        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        CAMs = returnCAM(features_blobs[0], weight_softmax, [predicted[0]])

        # render the CAM and output
        print('output CAM.jpg for the top1 prediction: %s' % classes[predicted[0]])
        images = images.sub(0.5).div(0.5)
        img = images.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img = np.concatenate([img, img, img], axis=2)

        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (500, 500)), cv2.COLORMAP_JET)

        result = heatmap * 0.2 + img * 0.8
        # print(result)
        cv2.imwrite('CAM/%d_final.jpg'%num, result)
        #cv2.imwrite('CAM/%d_norm.jpg' %num, img)

print('Finished whole test print')


for i in range(num_classes):
    print('Accuracy of %2s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))