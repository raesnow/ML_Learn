import matplotlib.pyplot as plt
from numpy import *


def load_dataset(filename, delim='\t'):
    with open(filename, 'r') as fr:
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
    dat_arr = [list(map(float, line)) for line in string_arr]
    return mat(dat_arr)


def pca(data_mat, top_n_feat=9999999):
    # 去平均值
    mean_vals = mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals
    cov_mat = cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = linalg.eig(mat(cov_mat))
    # 从小到大对N个值排序
    eig_val_ind = argsort(eig_vals)
    eig_val_ind = eig_val_ind[: -(top_n_feat+1) : -1]
    red_eig_vects = eig_vects[:, eig_val_ind]
    # 将数据转换到新空间
    low_d_data_mat = mean_removed * red_eig_vects
    recon_mat = (low_d_data_mat * red_eig_vects.T) + mean_vals
    return low_d_data_mat, recon_mat


def replace_nan_with_mean():
    data_mat = load_dataset("resource/secom.data", ' ')
    num_feat = shape(data_mat)[1]
    for i in range(num_feat):
        # 计算所有非NaN的平均值
        mean_val = mean(data_mat[nonzero(~isnan(data_mat[:, i].A))[0], i])
        # 将所有NaN置为平均值
        data_mat[nonzero(isnan(data_mat[:, i].A))[0], i] = mean_val
    return data_mat


if __name__ == "__main__":
    # 1. 测试：
    # 1.1 pca
    # data_mat = load_dataset("resource/testSet.txt")
    # low_d_mat, recon_mat = pca(data_mat, 2)
    # print(shape(low_d_mat))

    # 1.2 绘图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    # ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    # plt.show()

    # 2. 示例：半导体制造数据降维
    data_mat = replace_nan_with_mean()
    mean_vals = mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals
    cov_mat = cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = linalg.eig(mat(cov_mat))
    print(eig_vals)


