import torch
import numpy as np
import torch.utils.data as Data


def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]

    return new_assign


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)

    return small_cubic_data


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, TOTAL_SIZEBG, total_indicesbg, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all_bg = gt[total_indicesbg]
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                  PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    all_data_bg = select_small_cubic(TOTAL_SIZEBG, total_indicesbg, whole_data,
                                     PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                    PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    # print('train_data.shape=', train_data.shape)
    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                   PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]
    # print('x_val.shape, y_val.shape=', x_val.shape, y_val.shape)

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)


    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)

    all_data_bg.reshape(all_data_bg.shape[0], all_data_bg.shape[1], all_data_bg.shape[2], INPUT_DIMENSION)
    all_tensor_data_bg = torch.from_numpy(all_data_bg).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label_bg = torch.from_numpy(gt_all_bg).type(torch.FloatTensor)
    torch_dataset_all_bg = Data.TensorDataset(all_tensor_data_bg, all_tensor_data_label_bg)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valida_iter = Data.DataLoader(
        dataset=torch_dataset_valida,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_iter_bg = Data.DataLoader(
        dataset=torch_dataset_all_bg,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_iter, valida_iter, test_iter, all_iter, all_iter_bg