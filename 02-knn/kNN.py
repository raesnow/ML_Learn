import os.path
import re

from numpy import *
import operator
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

matplotlib.use('TkAgg')


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    k-近邻算法
    :param in_x: 用于分类的输入向量
    :param data_set: 训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return:
    """

    # 计算距离
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set  # 使用tile重复生成和data_set相同shape的数组
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()  # 返回数据排序以后，index的列表，注意：不是数据本身的排序结果

    # 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    """

    :param filename: datingTestSet.txt
    :return:
    """
    df = pd.read_table(filename, sep='\t', header=None)
    df.columns = ['flight miles earnd per year',
                  'percentage of time spent playing video games',
                  'liters of ice cream consumed per week',
                  'label']
    return_df = df.drop(axis=1, columns=[df.columns[3]])

    class_label_vector = df[df.columns[3]].to_list()
    return return_df, class_label_vector


def draw(data_df, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scatter = ax.scatter(data_df[data_df.columns[1]], data_df[data_df.columns[2]], 15.0*array(labels), 15.0*array(labels))
    scatter = ax.scatter(data_df[data_df.columns[0]], data_df[data_df.columns[1]], 15.0 * array(labels),
                         15.0 * array(labels))
    legend1 = ax.legend(
        *scatter.legend_elements(fmt=ticker.FuncFormatter(lambda x, pos: dating_result_list[int(x / 15 - 1)])),
        loc="upper left", title="Likes")
    ax.add_artist(legend1)

    plt.xlabel(data_df.columns[0])
    plt.ylabel(data_df.columns[1])
    plt.show()


def anto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    m = dataset.shape[0]
    norm_dataset = dataset - tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    # 2.1) 读入数据
    dating_data_df, dating_labels = file2matrix('resource/datingTestSet2.txt')
    print("dating_data_df: \n{}".format(dating_data_df))
    print("dating_labels: \n{}".format(dating_labels))

    # 2.2) 绘散点图
    draw(dating_data_df, dating_labels)

    # 2.3) 归一化
    norm_dating_data, ranges, min_vals = anto_norm(dating_data_df.values)
    print("\nnorm_dating_data: \n{}".format(norm_dating_data))
    print("ranges: \n{}".format(ranges))
    print("min_vals: \n{}".format(min_vals))

    # 2.4) 测试
    ho_ratio = 0.05
    m = norm_dating_data.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        # 取前10%的数据作测试，其他数据作为数据集
        classifier_result = classify0(norm_dating_data[i, :], norm_dating_data[num_test_vecs:, :],
                                      dating_labels[num_test_vecs:], 3)
        print("the classifier came back with: {}, the real answer is： {}".format(
            classifier_result, dating_labels[i]
        ))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: {}".format(error_count / float(num_test_vecs)))


def classify_person(ff_miles, percent_tats, ice_cream):
    dating_data_df, dating_labels = file2matrix('resource/datingTestSet2.txt')
    norm_dating_data, ranges, min_vals = anto_norm(dating_data_df.values)
    classifier_result = classify0(array([ff_miles, percent_tats, ice_cream]), norm_dating_data, dating_labels, 3)
    return dating_result_list[int(classifier_result - 1)]


def img2vector(filename):
    return_vect = zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            line_str = f.readline()
            for j in range(32):
                return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def hand_writing_class_test(path):
    hw_labels = []
    training_path = os.path.join(path, 'trainingDigits')
    training_file_list = os.listdir(training_path)
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        hw_labels.append(get_number_from_filename(training_file_list[i]))
        training_mat[i, :] = img2vector(os.path.join(training_path, training_file_list[i]))

    test_path = os.path.join(path, 'testDigits')
    test_file_list = os.listdir(test_path)
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        class_num = get_number_from_filename(test_file_list[i])
        vector_under_test = img2vector(os.path.join(test_path, test_file_list[i]))
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: {}, the real answer is： {}".format(
            classifier_result, class_num
        ))
        if classifier_result != class_num:
            error_count += 1.0
    print("the total error rate is: {}".format(error_count / float(m_test)))


def get_number_from_filename(filename):
    number_group = re.search(r'(\d+)_(\d+).txt', filename, re.M | re.I)
    return int(number_group.group(1))


if __name__ == "__main__":
    # 1. 简单的示例
    # group, labels = create_dataset()
    # result = classify0([0, 0], group, labels, 3)
    # print(result)

    # 2. 约会网站配对
    dating_result_list = ['not at all', 'in small doses', 'in large doses']
    # dating_class_test()
    # 输入数值进行分类
    # print(classify_person(46052, 6.441871, 1.805124))

    # 3. 手写识别系统
    hand_writing_resource_path = 'resource/digits'
    hand_writing_class_test(hand_writing_resource_path)
