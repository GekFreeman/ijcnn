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

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["debug", "train"], required=True)
parser.add_argument("--sigma", type=float, required=True)
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
batch_size = 32
iteration = params[mode]["iteration"]
lr = 0.01
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/root/workspace/MFSAN/dataset/office31_raw_image/Original_images/"
source1_name = "webcam"
source2_name = "dslr"
target_name = "amazon"

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
source1_test_load = data_loader.load_testing(root_path, source1_name, batch_size, kwargs)

def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    
#     for i in range(1, iteration + 1):
#         model.train()
#         LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
#         if (i - 1) % 100 == 0:
#             print("learning rate: ", LEARNING_RATE)
#         optimizer = torch.optim.SGD([
#             {'params': model.sharedNet.parameters()},
#             {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.block1.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.block2.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.block3.parameters(), 'lr': LEARNING_RATE},
#         ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
#
#         try:
#             source_data, source_label = source1_iter.next()
#         except Exception as err:
#             source1_iter = iter(source1_loader)
#             source_data, source_label = source1_iter.next()
#         try:
#             target_data, __ = target_iter.next()
#         except Exception as err:
#             target_iter = iter(target_train_loader)
#             target_data, __ = target_iter.next()
#         if cuda:
#             source_data, source_label = source_data.cuda(), source_label.cuda()
#             target_data = target_data.cuda()
#         source_data, source_label = Variable(source_data), Variable(source_label)
#         target_data = Variable(target_data)
#         optimizer.zero_grad()
#
#         cls_loss, mmd_loss, sparse_loss1, sparse_loss2, sparse_loss3, consolidate_loss = model(source_data, target_data, source_label)
#         gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
#         loss = cls_loss + gamma * mmd_loss + sparse_loss1 + sparse_loss2 + sparse_loss3
# #         loss = cls_loss + gamma * mmd_loss
#         loss.backward()
#         optimizer.step()
#
#         sparse_loss1_v = sparse_loss1 if sparse_loss1 == 0 else sparse_loss1.item()
#         sparse_loss2_v = sparse_loss2 if sparse_loss2 == 0 else sparse_loss2.item()
#         sparse_loss3_v = sparse_loss3 if sparse_loss3 == 0 else sparse_loss3.item()
#
#
#         if i % log_interval == 0:
#             print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tSparse_Loss1: {:.6f}\tSparse_Loss2: {:.6f}\tSparse_Loss3: {:.6f}\t'.format(
#                 i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), sparse_loss1_v, sparse_loss2_v, sparse_loss3_v))
#
#         if i % (log_interval * params[mode]["testing"]) == 0:
#             t_correct = test(model, stage=1)
#             if t_correct > correct:
#                 correct = t_correct
#             print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

    model.load_state_dict(torch.load("/root/workspace/MFSAN/exps_R/MFSAN_csrc23_R_4_A/tmp_results/stage-1-best.pt"))
    model.eval()
    # forget = memorys(model, stage=1)
    forget = np.load("/root/workspace/MFSAN/exps_R/MFSAN_csrc23_R_4_A/tmp_results/forget.npy")
    # np.save("/root/workspace/MFSAN/exps_R/MFSAN_csrc23_R_4_A/tmp_results/forget_source1.npy", forget)
    d = {}
    for i in range(256):
        mask = torch.ones((1, 256, 1, 1))
        mask[0][i][0][0] = 0
        mask = mask.cuda()
        remember = memory(model, stage=3, mask=mask)
        remember = np.argwhere(remember == 1).flatten()
        d[i] = len(np.setdiff1d(forget, remember))
    np.save("./tmp_results/mask_forgetting_b3.npy", d)

    # stage_1 = memory(model, stage=1)
    # np.save("./tmp_results/stage-1-memo-copy.npy", stage_1)
    # # rank
    # model.eval()
    # feature_map = {}
    # for data, target in source1_loader_imp:
    #     if cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data), Variable(target)
    #     optimizer.zero_grad()
    #     _, fm = model(data, 0, target, stage=-1)
    #     if "block1" in feature_map:
    #         feature_map["block1"] = np.concatenate([feature_map["block1"], fm["block1"]], axis=0)
    #         feature_map["block2"] = np.concatenate([feature_map["block2"], fm["block2"]], axis=0)
    #         feature_map["block3"] = np.concatenate([feature_map["block3"], fm["block3"]], axis=0)
    #     else:
    #         feature_map = fm
    # np.save("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/MFSAN_csrc23/exps_R/MFSAN_csrc23_R_4/tmp_results/s1_block1_featuremap.npy", feature_map["block1"])
    # np.save("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/MFSAN_csrc23/exps_R/MFSAN_csrc23_R_4/tmp_results/s1_block2_featuremap.npy", feature_map["block2"])
    # np.save("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/MFSAN_csrc23/exps_R/MFSAN_csrc23_R_4/tmp_results/s1_block3_featuremap.npy", feature_map["block3"])
    
    
    # tfeature_map = {}
    # for data, target in target_loader_imp:
    #     if cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data), Variable(target)
    #     optimizer.zero_grad()
    #     _, fm = model(data, 0, target, stage=-1)
    #     if "block1" in tfeature_map:
    #         tfeature_map["block1"] = np.concatenate([tfeature_map["block1"], fm["block1"]], axis=0)
    #         tfeature_map["block2"] = np.concatenate([tfeature_map["block2"], fm["block2"]], axis=0)
    #         tfeature_map["block3"] = np.concatenate([tfeature_map["block3"], fm["block3"]], axis=0)
    #     else:
    #         tfeature_map = fm
    # np.save("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/MFSAN_csrc23/exps_R/MFSAN_csrc23_R_4/tmp_results/t_block1_featuremap.npy", tfeature_map["block1"])
    # np.save("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/MFSAN_csrc23/exps_R/MFSAN_csrc23_R_4/tmp_results/t_block2_featuremap.npy", tfeature_map["block2"])
    # np.save("/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/MFSAN_csrc23/exps_R/MFSAN_csrc23_R_4/tmp_results/t_block3_featuremap.npy", tfeature_map["block3"])
    
    # s1_b1 = torch.tensor(feature_map["block1"]).cuda()
    # s1_b1 = torch.mean(s1_b1, dim=0)
    #
    # t_b1 = torch.tensor(tfeature_map["block1"]).cuda()
    # t_b1 = torch.mean(t_b1, dim=0)
    #
    # s1_b1 = s1_b1.reshape(1024, -1)
    # t_b1 = t_b1.reshape(1024, -1)
    #
    # st_b1 = torch.mm(s1_b1, t_b1.permute(1, 0))
    # s1_b1_norm = torch.sqrt(torch.sum(torch.pow(s1_b1, 2), dim=1)).unsqueeze(-1)
    # t_b1_norm = torch.sqrt(torch.sum(torch.pow(t_b1, 2), dim=1)).unsqueeze(0)
    # st_b1_norm = torch.mm(s1_b1_norm, t_b1_norm)
    # mask = torch.diag(st_b1 / (st_b1_norm + 1e-10))
    #
    # dm = {}
    # for i in range(1, 1023):
    #     tmask = 1 - (mask > torch.sort(mask)[0][-i]).reshape(1, mask.shape[0], 1, 1).float()
    #     t_correct = test(model, stage=3, mask=tmask)
    #     dm[i] = t_correct.item()
    # np.save("tmp_results/mask.npy", dm)
        
#     mask1 = 1 - (mask <= torch.sort(mask)[0][256]).reshape(1, mask.shape[0], 1, 1).float()
#     mask2 = 1 - (mask > torch.sort(mask)[0][-256]).reshape(1, mask.shape[0], 1, 1).float()
    
#     print("mask min:")
#     t_correct = test(model, stage=3, mask=mask1)
#     if t_correct > correct:
#         correct = t_correct
#     print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
    
#     print("mask max:")
#     t_correct = test(model, stage=3, mask=mask2)
#     if t_correct > correct:
#         correct = t_correct
#     print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
#     correct = 0
#     for i in range(1, iteration + 1):
#         model.train()
#         LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
#         if (i - 1) % 100 == 0:
#             print("learning rate: ", LEARNING_RATE)
#         optimizer = torch.optim.SGD([
#             {'params': model.sharedNet.parameters()},
#             {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.block1.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.block2.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.block3.parameters(), 'lr': LEARNING_RATE},
#         ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
#
#         try:
#             source_data, source_label = source2_iter.next()
#         except Exception as err:
#             source2_iter = iter(source2_loader)
#             source_data, source_label = source2_iter.next()
#         try:
#             target_data, __ = target_iter.next()
#         except Exception as err:
#             target_iter = iter(target_train_loader)
#             target_data, __ = target_iter.next()
#         if cuda:
#             source_data, source_label = source_data.cuda(), source_label.cuda()
#             target_data = target_data.cuda()
#         source_data, source_label = Variable(source_data), Variable(source_label)
#         target_data = Variable(target_data)
#         optimizer.zero_grad()
#
#         cls_loss, mmd_loss, sparse_loss1, sparse_loss2, sparse_loss3, consolidate_loss = model(source_data, target_data, source_label)
#         gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
#         loss = cls_loss + gamma * mmd_loss + sparse_loss1 + sparse_loss2 + sparse_loss3
# #         loss = cls_loss + gamma * mmd_loss
#         loss.backward()
#         optimizer.step()
#
#         sparse_loss1_v = sparse_loss1 if sparse_loss1 == 0 else sparse_loss1.item()
#         sparse_loss2_v = sparse_loss2 if sparse_loss2 == 0 else sparse_loss2.item()
#         sparse_loss3_v = sparse_loss3 if sparse_loss3 == 0 else sparse_loss3.item()
#
#
#         if i % log_interval == 0:
#             print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tSparse_Loss1: {:.6f}\tSparse_Loss2: {:.6f}\tSparse_Loss3: {:.6f}\t'.format(
#                 i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), sparse_loss1_v, sparse_loss2_v, sparse_loss3_v))
#
#         if i % (log_interval * params[mode]["testing"]) == 0:
#             t_correct = test(model, stage=1)
#             if t_correct > correct:
#                 correct = t_correct
#                 torch.save(model.state_dict(),
#                            "/root/workspace/MFSAN/exps_R/MFSAN_csrc23_R_4_A/tmp_results/stage-2-best-1w.pt")
#             print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

    # model.load_state_dict(torch.load("/root/workspace/MFSAN/exps_R/MFSAN_csrc23_R_4_A/tmp_results/stage-2-best-1w.pt"))
    # stage_2 = memory(model, stage=1)
    # np.save("./tmp_results/stage-2-memo-1w.npy", stage_2)

def test(model, stage=1, mask=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
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

def memorys(model, stage=1, mask=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    result = np.array([])
    with torch.no_grad():
        for data, target in source1_test_load:
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

        test_loss /= len(source1_test_load.dataset)
        print(source1_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(source1_test_load.dataset),
            100. * correct / len(source1_test_load.dataset)))
        print('\nsource1 accnum {}'.format(correct1))
    return result

if __name__ == '__main__':
    model = models.MFSAN(num_classes=31, sigma=args.sigma)
    # print(model)
    if cuda:
        model.cuda()
    train(model)
