import pickle
import torch


if __name__ == "__main__":
    threshold, epsilon = 0.01, 0.01
    with open(f'sparse/scipy_{threshold}_{epsilon}.pickle', 'rb') as data:
        r_matrix_list = pickle.load(data)
    new_matrix_list = []
    for i in range(len(r_matrix_list)):
        new_matrix_list.append(torch.tensor(r_matrix_list[i].todense()).to_sparse())
    torch.save(new_matrix_list, f'sparse/torch_{threshold}_{epsilon}.ckpt')
