import random

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def load_dataset():
    data_mat = []
    label_mat = []

    with open('resource/testSet.txt', 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()

    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights += alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(wei):
    weights = wei.getA()
    data_mat, label_mat = load_dataset()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 最佳拟合直线
    x = arange(-3.0, 3.0, 0.1)
    print(weights)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * array(data_matrix[i])
    return weights.reshape((3, 1))


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01

            # 随机选择更新，减少周期性的波动
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * array(data_matrix[rand_index])
            del(data_index[rand_index])
    return weights.reshape((n, 1))


def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('resource/horseColicTraining.txt')
    fr_test = open('resource/horseColicTest.txt')
    training_set = []
    training_labels = []

    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent1(array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = float(error_count) / num_test_vec
    print(f"the error rate of this test is {error_rate}")
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print(f"after {num_tests} iterations the average error rate is {error_sum/float(num_tests)}")


if __name__ == "__main__":
    # 1. 测试
    # data_arr, label_mat = load_dataset()
    # result = stoc_grad_ascent1(data_arr, label_mat)
    # print(result)
    # plot_best_fit(mat(result))

    # 2. 从疝气病预测病马的死亡率
    multi_test()
