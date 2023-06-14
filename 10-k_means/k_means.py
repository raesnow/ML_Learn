import matplotlib
import matplotlib.pyplot as plt
from numpy import *


def load_dataset(file_name):
    data_mat = []
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)
    return data_mat


def dist_eclud(veca, vecb):
    return sqrt(sum(power(veca - vecb, 2)))


def rand_cent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k, n)))

    for j in range(n):
        minj = min(dataset[:, j])
        rangej = float(max(dataset[:, j]) - minj)
        centroids[:, j] = minj + rangej * random.rand(k, 1)
    return centroids


def k_means(dataset, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))
    centroids = create_cent(dataset, k)
    cluster_changed = True

    while cluster_changed:
        cluster_changed = False

        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(centroids[j, :], dataset[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j

            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2

        print(centroids)
        for cent in range(k):
            pts_in_clust = dataset[nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(pts_in_clust, axis=0)
    return centroids, cluster_assment


def bi_k_means(dataset, k, dist_meas=dist_eclud):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))

    # 创建一个初始簇
    centroid0 = mean(dataset, axis=0).tolist()[0]
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(mat(centroid0), dataset[j, :]) ** 2

    while len(cent_list) < k:
        lowest_sse = inf

        for i in range(len(cent_list)):
            # 尝试划分每一簇
            pts_in_curr_cluster = dataset[nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_clust_ass = k_means(pts_in_curr_cluster, 2, dist_meas)
            sse_split = sum(split_clust_ass[:, 1])
            sse_not_split = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print("sse_split: {}, and not split: {}", sse_split, sse_not_split)

            if (sse_split + sse_not_split) < lowest_sse:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split
        # 更新簇的分配结果
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        print("the best_cent_to_split is: {}".format(best_cent_to_split))
        print("the len of best_clust_ass is: {}".format(len(best_clust_ass)))
        cent_list[best_cent_to_split] = best_new_cents[0, :].A[0]
        cent_list.extend(best_new_cents[1, :].A)
        cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
    return mat(cent_list), cluster_assment


def dist_slc(veca, vecb):
    a = sin(veca[0, 1] * pi / 180) * sin(vecb[0, 1] * pi /180)
    b = cos(veca[0, 1] * pi / 180) * cos(vecb[0, 1] * pi /180) * cos(pi * (vecb[0, 0] - veca[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def cluster_clubs(num_clust=5):
    dat_list = []
    with open("resource/places.txt", 'r') as fr:
        for line in fr.readlines():
            line_arr = line.split('\t')
            dat_list.append([float(line_arr[4]), float(line_arr[3])])

    dat_mat = mat(dat_list)
    my_centroids, clust_assing = bi_k_means(dat_mat, num_clust, dist_meas=dist_slc)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    img_p = plt.imread("resource/Portland.png")
    ax0.imshow(img_p)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)

    for i in range(num_clust):
        pts_in_curr_cluster = dat_mat[nonzero(clust_assing[:, 0].A == i)[0], :]
        marker_style = scatter_markers[i % len(scatter_markers)]
        ax1.scatter(pts_in_curr_cluster[:, 0].flatten().A[0], pts_in_curr_cluster[:, 1].flatten().A[0],
                    marker=marker_style, s=90)
    ax1.scatter(my_centroids[:, 0].flatten().A[0], my_centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == "__main__":
    # 1. 测试
    # 1.1 k-means
    # data_mat = mat(load_dataset("resource/testSet.txt"))
    # my_centroids, clust_assing = k_means(data_mat, 4)
    # print("######## Result ########")
    # print(my_centroids)
    # 1.2 bisecting k-means
    # data_mat2 = mat(load_dataset("resource/testSet2.txt"))
    # my_centroids2, clust_assing2 = bi_k_means(data_mat2, 3)
    # print("######## Result ########")
    # print(my_centroids2)

    # 2. 示例：对地图上的点进行聚类
    cluster_clubs(6)
