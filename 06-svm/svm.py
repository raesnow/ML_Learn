import random
from numpy import *
from os import listdir, path


def load_dataset(filename):
    data_mat = []
    label_mat = []
    with open(filename) as f:
        for line in f.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter = 0

    while iter < max_iter:
        alpha_pairs_changes = 0
        for i in range(m):
            f_xi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            ei = f_xi - float(label_mat[i])
            if (label_mat[i] * ei < -toler and alphas[i] < c) or (label_mat[i] * ei > toler and alphas[i] > 0):
                j = select_jrand(i, m)
                f_xj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                ej = f_xj - float(label_mat[j])

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, c + alphas[j] + alphas[i])
                if l == h:
                    print("L == H")
                    continue

                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T \
                      - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= label_mat[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not move enough")
                    continue

                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] and c > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and c > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changes += 1
                print(f"iter: {iter}, i: {i}, pairs changed {alpha_pairs_changes}")

        if alpha_pairs_changes == 0:
            iter += 1
        print(f"iteration number: {iter}")
    return b, alphas


class OptStruct:
    def __init__(self, data_mat_in, class_labels, c, toler, k_tup):
        self.x = data_mat_in
        self.label_mat = class_labels
        self.c = c
        self.tol = toler
        self.m = shape(data_mat_in)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.e_cache = mat(zeros((self.m, 2)))
        self.k = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernel_trans(self.x, self.x[i, :], k_tup)


def calc_ek(os: OptStruct, k):
    f_xk = float(multiply(os.alphas, os.label_mat).T * os.k[:, k]) + os.b
    ek = f_xk - float(os.label_mat[k])
    return ek


def select_j(i, os: OptStruct, ei):
    max_k = -1
    max_delta_e = 0
    ej = 0

    if shape(ei) == (1, 1):
        os.e_cache[i] = [1, ei[0, 0]]
    else:
        os.e_cache[i] = [1, ei]
    valid_e_cache_list = nonzero(os.e_cache[:, 0].A)[0]
    if len(valid_e_cache_list) > 1:
        for k in valid_e_cache_list:
            if k == i:
                continue
            ek = calc_ek(os, k)
            delta_e = abs(ei - ek)
            # 选择具有最大步长的j
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    else:
        j = select_jrand(i, os.m)
        ej = calc_ek(os, j)
    return j, ej


def update_ek(os, k):
    ek = calc_ek(os, k)
    if shape(ek) == (1, 1):
        os.e_cache[k] = [1, ek[0, 0]]
    else:
        os.e_cache[k] = [1, ek]


def inner_l(i, os):
    ei = calc_ek(os, i)
    if (os.label_mat[i] * ei < -os.tol and os.alphas[i] < os.c) or (os.label_mat[i] * ei > os.tol and os.alphas[i] > 0):
        j, ej = select_j(i, os, ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.label_mat[i] != os.label_mat[j]:
            l = max(0, os.alphas[j] - os.alphas[i])
            h = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            l = max(0, os.alphas[j] + os.alphas[i] - os.c)
            h = min(os.c, os.c + os.alphas[j] + os.alphas[i])
        if l == h:
            print("l == h")
            return 0

        eta = 2.0 * os.k[i, j] - os.k[i, i] - os.k[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        os.alphas[j] -= os.label_mat[j] * (ei - ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], h, l)
        # 更新误差缓存
        update_ek(os, j)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0

        os.alphas[i] += os.label_mat[j] * os.label_mat[i] * (alpha_j_old - os.alphas[j])
        update_ek(os, i)

        b1 = os.b - ei - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.k[i, i] -\
             os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.k[i, j]
        b2 = os.b - ej - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.k[i, j] -\
             os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.k[j, j]
        if 0 < os.alphas[i] and os.c > os.alphas[i]:
            os.b = b1
        elif 0 < os.alphas[j] and os.c > os.alphas[j]:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    return 0


def smo_p(data_mat_in, class_labels, c, toler, max_iter, k_tup=('lin', 0)):
    os = OptStruct(mat(data_mat_in), mat(class_labels).transpose(), c, toler, k_tup)
    iter = 0
    entire_set = True
    alpha_pairs_changes = 0
    while iter < max_iter and (alpha_pairs_changes > 0 or entire_set):
        alpha_pairs_changes = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changes += inner_l(i, os)
                print(f"fullSet, iter: {iter}, i: {i}, pairs changed: {alpha_pairs_changes}")
            iter += 1
        else:
            non_bound_is = nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            for i in non_bound_is:
                alpha_pairs_changes += inner_l(i, os)
                print(f"non-bound, iter: {iter}, i: {i}, pairs changed: {alpha_pairs_changes}")
            iter += 1

        if entire_set:
            entire_set = False
        elif alpha_pairs_changes == 0:
            entire_set = True
        print(f"iteration number: {iter}")
    return os.b, os.alphas


def calc_ws(alphas, data_arr, class_labels):
    x = mat(data_arr)
    label_mat = mat(class_labels).transpose()
    m, n = shape(x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


def kernel_trans(x, a, k_tup):
    m, n = shape(x)
    k = mat(zeros((m, 1)))
    if k_tup[0] == 'lin':
        k = x * a.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = x[j, :] - a
            k[j] = delta_row * delta_row.T
        k = exp(k / (-1 * k_tup[1] ** 2))
    else:
        raise NameError("the kernel is not recognized!")
    return k


def test_rbf(k1=1.3):
    data_arr, label_arr = load_dataset("resource/testSetRBF.txt")
    b, alphas = smo_p(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))

    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    s_vs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    print(f"there are {shape(s_vs)[0]} Support Vectors")

    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print(f"the training error rate is {float(error_count) / m}")

    data_arr, label_arr = load_dataset("resource/testSetRBF2.txt")
    error_count = 0
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print(f"the test error rate is {float(error_count) / m}")


def img2vector(filename):
    return_vect = zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            line_str = f.readline()
            for j in range(32):
                return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def load_images(dir_name):
    hw_labels = []
    training_file_list = listdir(dir_name)
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        training_mat[i, :] = img2vector(path.join(dir_name, file_name_str))
    return training_mat, hw_labels


def test_digits(k_tup=('rbf', 10)):
    data_arr, label_arr = load_images('../02-knn/resource/digits/trainingDigits')
    b, alphas = smo_p(data_arr, label_arr, 200, 0.0001, 10000, k_tup)

    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    s_vs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    print(f"there are {shape(s_vs)[0]} Support Vectors")

    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print(f"the training error rate is {float(error_count) / m}")

    data_arr, label_arr = load_images('../02-knn/resource/digits/testDigits')
    error_count = 0
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print(f"the test error rate is {float(error_count) / m}")


if __name__ == "__main__":
    # 1. 简单版SMO
    # data, labels = load_dataset("resource/testSet.txt")
    # b, alphas = smo_simple(data, labels, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas>0])
    #
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(data[i], labels[i])

    # 2. 完整版SMO
    # b, alphas = smo_p(data, labels, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas>0])
    # print(calc_ws(alphas, data, labels))

    # 3. 核函数的使用
    # test_rbf(k1=0.1)

    # 4. 手写识别问题
    test_digits()
