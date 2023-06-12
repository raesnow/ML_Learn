from bs4 import BeautifulSoup
from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []

    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regres(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T * x_mat

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye((m)))

    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T / (-2.0 * k**2))
    xTx = x_mat.T * (weights * x_mat)

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = shape(test_arr)[0]
    y_hat = zeros(m)

    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regres(x_mat, y_mat, lam=0.2):
    xTx = x_mat.T * x_mat
    denom = xTx + eye(shape(x_mat)[1]) * lam

    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    # 数据标准化
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var

    num_test_pts = 30
    w_mat = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regres(x_mat, y_mat, exp(i-10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):
    in_mat = x_mat.copy()
    in_means = mean(in_mat,0)
    in_var = var(in_mat,0)
    in_mat = (in_mat - in_means)/in_var
    return in_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)

    m, n = shape(x_mat)
    return_mat = zeros((num_it, n))
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()

    for i in range(num_it):
        lowest_error = inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps*sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


# 参照 https://blog.csdn.net/u011629133/article/details/52296892
def scrapePage(ret_x, ret_y, in_file, yr, num_pce, orig_prc):
    with open(in_file, 'r', encoding="utf-8") as fr:
        soup = BeautifulSoup(fr.read())

    i = 1
    current_row = soup.findAll("table", r=str(i))
    while len(current_row) > 0:
        title = current_row[0].findAll("a")[1].text.lower()

        # 查找是否有全新标签
        new_flag = 0.0
        if "new" in title or "nisb" in title:
            new_flag = 1.0

        sold_unicode = current_row[0].findAll("td")[3].findAll("span")
        if len(sold_unicode) == 0:
            print("item {} not sell".format(i))
        else:
            # 解析页面获取当前价格
            sold_price = current_row[0].findAll("td")[4]
            price_str = sold_price.text.replace('$','').replace(',','')
            if len(sold_price) > 1:
                price_str = price_str.replace("Free shipping", "")
            selling_price = float(price_str)

            # 去掉不完整的套装价格
            if selling_price > orig_prc * 0.5:
                ret_x.append([yr, num_pce, new_flag, orig_prc])
                ret_y.append(selling_price)
        i += 1
        current_row = soup.findAll("table", r=str(i))


def set_data_collect(ret_x, ret_y):
    scrapePage(ret_x, ret_y, 'resource/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(ret_x, ret_y, 'resource/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(ret_x, ret_y, 'resource/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(ret_x, ret_y, 'resource/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(ret_x, ret_y, 'resource/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(ret_x, ret_y, 'resource/setHtml/lego10196.html', 2009, 3263, 249.99)


def cross_validation(x_arr, y_arr, num_val=10):
    m = len(y_arr)
    index_list = list(range(m))
    error_mat = zeros((num_val, 30))

    for i in range(num_val):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        random.shuffle(index_list)

        for j in range(m):
            if j < m * 0.9:
                train_x.append(x_arr[index_list[j]])
                train_y.append(y_arr[index_list[j]])
            else:
                test_x.append(x_arr[index_list[j]])
                test_y.append(y_arr[index_list[j]])

        w_mat = ridge_test(train_x, train_y)
        for k in range(30):
            mat_testx = mat(test_x)
            mat_trainx = mat(train_x)
            mean_train = mean(mat_trainx, 0)
            var_train = var(mat_trainx, 0)
            mat_testx = (mat_testx - mean_train) / var_train
            y_est = mat_testx * mat(w_mat[k, :]).T + mean(train_y)
            error_mat[i, k] = rss_error(y_est.T.A, array(test_y))

    mean_errors = mean(error_mat, 0)
    min_mean = float(min(mean_errors))
    best_weights = w_mat[nonzero(mean_errors==min_mean)]
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    mean_x = mean(x_mat, 0)
    var_x = var(x_mat, 0)
    un_reg = best_weights / var_x
    print("the best model from ridge regression is {}".format(un_reg))
    print("with constant term: {}".format(-1 * sum(multiply(mean_x, un_reg)) + mean(y_mat)))


if __name__ == "__main__":
    # 1. 测试
    # 1.1 计算回归系数
    # x_arr, y_arr = load_data_set('resource/ex0.txt')
    # ws = stand_regres(x_arr, y_arr)
    # print(ws)

    # 1.2 绘制数据集散点图和最佳拟合直线图
    # x_mat = mat(x_arr)
    # y_mat = mat(y_arr)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    # x_copy = x_mat.copy()
    # x_copy.sort(0)
    # y_hat = x_copy * ws
    # ax.plot(x_copy[:, 1], y_hat)
    # plt.show()

    # 1.3 计算预测值和真实值的相关性
    # print(corrcoef((x_mat * ws).T, y_mat))

    # 2. LWLR
    # 2.1 计算
    # x_arr, y_arr = load_data_set('resource/ex0.txt')
    # y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.003)

    # 2.2 绘图
    # x_mat = mat(x_arr)
    # srt_ind = x_mat[:, 1].argsort(0)
    # x_sort = x_mat[srt_ind][:, 0, :]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x_sort[:, 1], y_hat[srt_ind])
    # ax.scatter(x_mat[:, 1].flatten().A[0], mat(y_arr).T.flatten().A[0], s=2, c='red')
    # plt.show()

    # 3. 示例：预测鲍鱼的年龄
    # 3.1 线性回归和LWLR
    # ab_x, ab_y = load_data_set('resource/abalone.txt')
    # y_hat01 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
    # y_hat1 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1)
    # y_hat10 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)
    # print("lwlr rss_error: 0.1: {}, 1: {}, 10: {}".format(
    #     rss_error(ab_y[0:99], y_hat01.T),
    #     rss_error(ab_y[0:99], y_hat1.T),
    #     rss_error(ab_y[0:99], y_hat10.T)
    # ))
    #
    # y_hat01_new = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 0.1)
    # y_hat1_new = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 1)
    # y_hat10_new = lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 10)
    # print("on new data, lwlr rss_error: 0.1: {}, 1: {}, 10: {}".format(
    #     rss_error(ab_y[100:199], y_hat01_new.T),
    #     rss_error(ab_y[100:199], y_hat1_new.T),
    #     rss_error(ab_y[100:199], y_hat10_new.T)
    # ))
    #
    # ws = stand_regres(ab_x[0:99], ab_y[0:99])
    # y_hat = mat(ab_x[100:199]) * ws
    # print("on new data, stand rss_error: {}".format(rss_error(ab_y[100:199], y_hat.T.A)))

    # 3.2 岭回归
    # ab_x, ab_y = load_data_set('resource/abalone.txt')
    # ridge_weights = ridge_test(ab_x, ab_y)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridge_weights)
    # plt.show()

    # 3.3 向前逐步回归
    # ab_x, ab_y = load_data_set('resource/abalone.txt')
    # weights = stage_wise(ab_x, ab_y, 0.001, 5000)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(weights)
    # plt.show()

    # 4. 示例：预测乐高玩具套装的价格
    # 4.1 获取数据
    lg_x = []
    lg_y = []
    set_data_collect(lg_x, lg_y)
    print(lg_x)
    print(lg_y)

    # 4.2 线性回归
    lg_x1 = mat(ones((63, 5)))
    lg_x1[:, 1:5] = mat(lg_x)
    ws = stand_regres(lg_x1, lg_y)
    print(ws)

    # 4.3 交叉验证
    cross_validation(lg_x, lg_y, 10)
