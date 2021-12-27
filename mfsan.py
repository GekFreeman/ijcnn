from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pdb
import argparse
import numpy as np
import mmd
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["debug", "train"], required=True)
parser.add_argument("--sigma", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args = parser.parse_args()

# Training settings
mode = args.mode # debug/train
params = {
    "debug": {
        "iteration": 10000,
        "testing": 10
    },
    "train": {
        "iteration": 10000,
        "testing": 10
    }
}
batch_size = args.batch_size
iteration = params[mode]["iteration"]
lr = 0.01
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/dataset/office31_raw_image/Original_images/"
source1_name = "amazon"
source2_name = "dslr"
target_name = "webcam"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
source1_loader_imp = data_loader.load_importance(root_path, source1_name, 1, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

source1_loader_imp = data_loader.load_importance(root_path, source1_name, 32, kwargs)
target_loader_imp = data_loader.load_importance(root_path, target_name, 32, kwargs)

def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.block1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.block2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.block3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, sparse_loss1, sparse_loss2, sparse_loss3, consolidate_loss = model(source_data, target_data, source_label)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * mmd_loss + sparse_loss1 + sparse_loss2 + sparse_loss3
#         loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer.step()

        sparse_loss1_v = sparse_loss1 if sparse_loss1 == 0 else sparse_loss1.item()
        sparse_loss2_v = sparse_loss2 if sparse_loss2 == 0 else sparse_loss2.item()
        sparse_loss3_v = sparse_loss3 if sparse_loss3 == 0 else sparse_loss3.item()


        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tSparse_Loss1: {:.6f}\tSparse_Loss2: {:.6f}\tSparse_Loss3: {:.6f}\t'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), sparse_loss1_v, sparse_loss2_v, sparse_loss3_v))

        if i % (log_interval * params[mode]["testing"]) == 0:
            t_correct = test(model, stage=1)
            if t_correct > correct:
                correct = t_correct
#                 torch.save(model.state_dict(),
#                            "/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/results_code/resnet101_512_512_256/adw/tmp_results/stage-1-best.pt")
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

#     model.load_state_dict(torch.load("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/results_code/resnet101_512_512_256/adw/tmp_results/stage-1-best.pt"))

    correct = 0
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.block1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.block2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.block3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, sparse_loss1, sparse_loss2, sparse_loss3, consolidate_loss = model(source_data, target_data, source_label)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * mmd_loss + sparse_loss1 + sparse_loss2 + sparse_loss3
#         loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer.step()

        sparse_loss1_v = sparse_loss1 if sparse_loss1 == 0 else sparse_loss1.item()
        sparse_loss2_v = sparse_loss2 if sparse_loss2 == 0 else sparse_loss2.item()
        sparse_loss3_v = sparse_loss3 if sparse_loss3 == 0 else sparse_loss3.item()


        if i % log_interval == 0:
            print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tSparse_Loss1: {:.6f}\tSparse_Loss2: {:.6f}\tSparse_Loss3: {:.6f}\t'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), sparse_loss1_v, sparse_loss2_v, sparse_loss3_v))

        if i % (log_interval * params[mode]["testing"]) == 0:
            t_correct = test(model, stage=1)
            if t_correct > correct:
                correct = t_correct
#                 torch.save(model.state_dict(),
#                            "/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/results_code/resnet101_512_512_256/adw/tmp_results/stage-2-best.pt")
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test(model, stage=1, mask1=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, _ = model(data, mask1=mask1, stage=stage)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            
            pred = pred1
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}'.format(correct1))
    return correct


def memory(model, stage=1, mask=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    result = np.array([])
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, _ = model(data, mask=mask, stage=stage)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)

            pred = pred1
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()

            result = np.concatenate([result, pred.eq(target.data.view_as(pred)).double().cpu().numpy()])

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}'.format(correct1))
    return result



if __name__ == '__main__':
    model = models.MFSAN(num_classes=31, sigma=args.sigma)
    # print(model)
    if cuda:
        model.cuda()
    train(model)
