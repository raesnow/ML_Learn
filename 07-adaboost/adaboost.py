from numpy import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def load_simple_data():
    data_mat = matrix([
        [1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]
    ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    ret_array = ones((shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, d):
    data_matrix = mat(data_arr)
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_clas_est = mat(zeros((m, 1)))
    min_error = inf

    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps

        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)

                err_arr = mat(ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = d.T * err_arr
                print(f"split: dim {i}, thresh: {thresh_val}, thresh inequal: {inequal}, the weighted error: {weighted_error}")

                if weighted_error < min_error:
                    min_error = weighted_error
                    best_clas_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_clas_est


def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    weak_class_arr = []
    m = shape(data_arr)[0]
    d = mat(ones((m, 1)) / m)
    agg_class_est = mat(zeros((m, 1)))

    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, d)

        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)

        expon = multiply(-1 * alpha * mat(class_labels).T, class_est)
        d = multiply(d, exp(expon))
        d = d / d.sum()

        agg_class_est += alpha * class_est
        print(f"agg_class_est: {agg_class_est}")
        agg_errors = multiply(sign(agg_class_est) != mat(class_labels).T, ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print(f"total error: {error_rate}")
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def ada_classify(dat_to_class, classifier_arr):
    data_matrix = mat(dat_to_class)
    m = shape(data_matrix)[0]
    agg_class_est = mat(zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)
    return sign(agg_class_est)


def load_dataset(filename):
    num_feat = len(open(filename).readline().split('\t'))
    data_mat = []
    label_mat = []
    with open(filename) as f:
        for line in f.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat - 1):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def plot_roc(pred_strengths, class_labels):
    cur = (1.0, 1.0)
    y_sum = 0.0
    num_pos_clas = sum(array(class_labels) == 1.0)
    y_step = 1 / float(num_pos_clas)
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    sorted_indicies = pred_strengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.axis([0, 1, 0, 1])
    plt.show()

    print(f"the Area Under the Curve is: {y_sum * x_step}")


if __name__ == "__main__":
    # 1. 测试数据
    # data_mat, class_labels = load_simple_data()
    # d = mat(ones((5, 1)) / 5)
    # print(build_stump(data_mat, class_labels, d))

    # classifier_arr = ada_boost_train_ds(data_mat, class_labels, 9)
    # print(ada_classify([0, 0], classifier_arr))
    # print(ada_classify([[5, 5], [0, 0]], classifier_arr))

    # 2. 预测患有疝气病的马是否能够存活
    data_arr, label_arr = load_dataset('resource/horseColicTraining2.txt')
    classifier_arr, agg_class_est = ada_boost_train_ds(data_arr, label_arr, 50)
    test_arr, test_label_arr = load_dataset('resource/horseColicTest2.txt')
    prediction10 = ada_classify(test_arr, classifier_arr)
    err_arr = mat(ones((67, 1)))
    print(err_arr[prediction10 != mat(test_label_arr).T].sum() / 67.0)

    # 绘ROC图
    plot_roc(agg_class_est.T, label_arr)
