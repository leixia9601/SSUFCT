import math
import time
import collections
import numpy as np
import scipy.io as sio
import os
import copy
import pynvml
import torch
import record
import geniter
import tools
import get_cls_map
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from Utils import SSUFCT_model
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]

global Dataset
dataset = 'HT2013'
Dataset = dataset.upper()


def load_dataset(Dataset):
    if Dataset == 'IN':
        mat_data = sio.loadmat('./datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('./datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.90

        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'HHK':
        GF5_HHK = sio.loadmat('./datasets/GF5_HHK.mat')
        gt_HHK = sio.loadmat('./datasets/GF5_HHK_gt.mat')
        data_hsi = GF5_HHK['data']
        gt_hsi = gt_HHK['label']
        TOTAL_SIZE = 285054
        VALIDATION_SPLIT = 0.85
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'HT2013':
        Ht13 = sio.loadmat('./datasets/Houston2013.mat')
        gt_Ht13 = sio.loadmat('./datasets/Houston2013_gt.mat')
        data_hsi = Ht13['Houston']
        gt_hsi = gt_Ht13['Houston_gt']
        TOTAL_SIZE = 15029
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def get_memory_used_MB():

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_info_MB = mem_info.used // 1024 ** 2

    return mem_info_MB


def MBP(samples, row_top):

    size = samples[0].shape
    mean = np.zeros(size)
    for s in samples:
        mean = mean + s

    mean /= float(len(samples))

    cov_row = np.zeros((samples.shape[2], samples.shape[2]))
    for s in samples:
        diff = s - mean
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row /= float(len(samples))

    row_eval, row_evec = np.linalg.eig(cov_row)
    sorted_index = np.argsort(row_eval)
    X = row_evec[:, sorted_index[:-row_top - 1: -1]]
    Y = row_eval[sorted_index[:-row_top - 1: -1]]

    return X, Y, mean


init_GPU_memory = int(get_memory_used_MB())
print('init_GPU_memory is: ', init_GPU_memory)
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)

MBP_components = 10
X_MBP, Y, mean = MBP(data_hsi, MBP_components)
reduced_data = np.dot(data_hsi, X_MBP)
data_hsi = reduced_data

image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)


print('-----Importing Setting Parameters-----')
ITER = 1
PATCH_LENGTH = 4
lr, num_epochs, batch_size = 0.0003, 200, 64
model_name = 'Houston13_SSUFCT'
loss = torch.nn.CrossEntropyLoss()


img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_ratio = 1
VAL_SIZE = int(VAL_ratio * TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)), 'constant', constant_values=0)


def train(net, train_iter, valida_iter, loss, optimizer, device, epochs, model_name):

    beforetrain_memory = get_memory_used_MB()
    print("beforetrain_memory:", beforetrain_memory)
    loss_list = [100]
    best_model = None
    net = net.to(device)
    print("training on ", device)
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    GPU_memory_usage = []
    best_epoch = 0
    best_acc = 0

    for epoch in range(epochs):
        time_epoch = time.time()
        batch_count, train_l_sum, n, train_acc_sum = 0, 0, 0, 0
        for X, y in train_iter:
            net.train()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            aftertrain_memory = get_memory_used_MB()


        toc3 = time.time()
        train_GPU_memory = aftertrain_memory - beforetrain_memory
        GPU_memory_usage.append(train_GPU_memory)
        valida_acc, valida_loss = record.evaluate_accuracy(valida_iter, net, loss, device)

        if best_acc <= valida_acc:
            best_acc = valida_acc
            best_epoch = epoch + 1
            best_model = copy.deepcopy(net)

        loss_list.append(valida_loss)
        train_loss_list.append(train_l_sum / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss.cpu())
        valida_acc_list.append(valida_acc)


        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.2f sec, GPU_memory %d'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, valida_acc, toc3 - time_epoch,
               train_GPU_memory))

    weights_path = f'./weights'
    os.makedirs(weights_path, exist_ok=True)
    path = f'./weights/{dataset}_{model_name}_{best_epoch}.pth'
    torch.save(best_model.state_dict(), path)
    average_GPU_memory_used = int(sum(GPU_memory_usage) / len(GPU_memory_usage))
    print('average_GPU_memory_used is ' + str(average_GPU_memory_used) + ' MB')

    plt.title('Accuracy')
    plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green', label='Train Accuracy')
    plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue', label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    plt.title('Loss')
    plt.plot(np.linspace(1, epoch, len(train_loss_list)), train_loss_list, color='red', label='Train Loss')
    plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    plt.plot(np.linspace(1, epoch, len(GPU_memory_usage)), GPU_memory_usage)
    plt.xlabel("Epoch")
    plt.ylabel("GPU Memory Allocated (MB)")
    plt.title("GPU Memory Usage during Training (in MB)")
    plt.show()

    return best_epoch, best_model, average_GPU_memory_used


def sampling(proportion, ground_truth, bg=False):

    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth) + 1
    for i in range(m):
        a = 0 if bg == True else 1
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + a]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)

    return train_indexes, test_indexes


net = SSUFCT_model.CET(CLASSES_NUM)
"""
# Training
"""
for index_iter in range(ITER):

    print('iter:', index_iter)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)
    _, total_indicesbg = sampling(1, gt, bg=True)
    TOTAL_SIZEBG = len(total_indicesbg)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(VAL_ratio * TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')
    train_iter, valida_iter, test_iter, all_iter, all_iter_bg = \
        geniter.generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                              TOTAL_SIZEBG, total_indicesbg, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,
                              INPUT_DIMENSION, batch_size, gt)
    tic1 = time.time()
    best_epoch, best_model, average_GPU_memory_used = train(net, train_iter, valida_iter, loss, optimizer, device,
                                                            num_epochs, model_name)
    toc1 = time.time()

    path = f'./weights/{dataset}_{model_name}_{best_epoch}.pth'
    net.load_state_dict(torch.load(path))
    pred_test = []
    tic2 = time.time()
    with torch.no_grad():
        for X, y in test_iter:
            tic3 = time.time()
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
            toc3 = time.time()
    toc2 = time.time()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1

    # summary(net, input_size=(1, img_rows, img_cols, INPUT_DIMENSION))

    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

    target_names = [' Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
         'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', ' Parking Lot 2',
                    'Tennis Court', 'Running Track']
    classification = classification_report(gt_test[:-VAL_SIZE], pred_test, digits=4, target_names=target_names)
    classification = str(classification)

    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    print('Training Time: ', toc1 - tic1)
    print('TESTING TIME: ', toc2 - tic2)
    print('each_acc=', each_acc)
print('OA, AA, KAPPA: ', OA, AA, KAPPA)
average_OA = sum(OA) / len(OA)
average_AA = sum(AA) / len(AA)
average_KAPPA = sum(KAPPA) / len(KAPPA)
print('average_OA, average_AA, KAPPA =', average_OA, average_AA, average_KAPPA)

"""
# Map, Records
"""
print("--------" + " Training Finished-----------")

wholehsi_predict_time = tools.generate_png(all_iter_bg, best_model, gt_hsi, Dataset, device, total_indicesbg)
print('wholehsi_predict_time=', round(wholehsi_predict_time, 3))
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, confusion_matrix, classification,
                     average_OA, average_AA, average_KAPPA, wholehsi_predict_time, average_GPU_memory_used,
                     './record/' + dataset + '_SSUFCT' + str(img_rows) + '_' + Dataset + 'split：' + str(
                         VALIDATION_SPLIT) + 'lr：' + str(lr) + '_patch9x9' + '.txt')

gt_hsi_predict_time = get_cls_map.get_cls_map(net, device, all_iter, gt_hsi)
print('gt_hsi_predict_time=', round(gt_hsi_predict_time, 3))




