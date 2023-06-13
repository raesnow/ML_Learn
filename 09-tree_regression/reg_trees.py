from numpy import *

class TreeNode():
    def __init__(self, feat, val, right, left):
        feature_to_split_on = feat
        value_of_split = val
        right_branch = right
        left_branch = left


def load_dataset(file_name):
    data_mat = []
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)
    return data_mat


def bin_split_dataset(dataset, feature, value):
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :]
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(dataset):
    return mean(dataset[:, -1])


def reg_err(dataset):
    return var(dataset[:, -1]) * shape(dataset)[0]


def linear_solve(dataset):
    m, n = shape(dataset)
    x = mat(ones((m, n)))
    y = mat(ones((m, 1)))
    x[:, 1:n] = dataset[:, 0:n-1]
    y = dataset[:, -1]
    xTx = x.T * x

    if linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, cannot do inverse, try increasing the second value of ops")

    ws = xTx.I * (x.T * y)
    return ws, x, y


def model_leaf(dataset):
    ws, x, y = linear_solve(dataset)
    return ws


def model_err(dataset):
    ws, x, y = linear_solve(dataset)
    y_hat = x * ws
    return sum(power(y - y_hat, 2))


def choose_best_split(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(dataset)

    m, n = shape(dataset)
    s = err_type(dataset)
    best_s = inf
    best_index = 0
    best_value = 0
    for feat_index in range(n-1):
        for split_val in set(asarray(dataset[:, feat_index]).ravel()):
            mat0, mat1 = bin_split_dataset(dataset, feat_index, split_val)
            if shape(mat0)[0] < tol_n or shape(mat1)[0] < tol_n:
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s

    if (s - best_s) < tol_s:
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_dataset(dataset, best_index, best_value)
    if shape(mat0)[0] < tol_n or shape(mat1)[0] < tol_n:
        return None, leaf_type(dataset)
    return best_index, best_value


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat is None:
        return val

    ret_tree = {}
    ret_tree["sp_ind"] = feat
    ret_tree["sp_val"] = val
    lset, rset = bin_split_dataset(dataset, feat, val)
    ret_tree["left"] = create_tree(lset, leaf_type, err_type, ops)
    ret_tree["right"] = create_tree(rset, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return type(obj).__name__ == "dict"


def get_mean(tree):
    if is_tree(tree["right"]):
        tree["right"] = get_mean(tree["right"])
    if is_tree(tree["left"]):
        tree["left"] = get_mean(tree["left"])
    return (tree["left"] + tree["right"]) / 2.0


def prune(tree, test_data):
    if shape(test_data)[0] == 0:
        return get_mean(tree)

    if is_tree(tree["right"]) or is_tree(tree["left"]):
        lset, rset = bin_split_dataset(test_data, tree["sp_ind"], tree["sp_val"])
    if is_tree(tree["left"]):
        tree["left"] = prune(tree["left"], lset)
    if is_tree(tree["right"]):
        tree["right"] = prune(tree["right"], rset)

    if not is_tree(tree["left"]) and not is_tree(tree["right"]):
        lset, rset = bin_split_dataset(test_data, tree["sp_ind"], tree["sp_val"])
        error_no_merge = sum(power(lset[:, -1] - tree["left"], 2)) + sum(power(rset[:, -1] - tree["right"], 2))
        tree_mean = (tree["left"] + tree["right"]) / 2.0
        error_merge = sum(power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print("merging")
            return tree_mean
    return tree


def reg_tree_eval(model, in_dat):
    return float(model)


def model_tree_eval(model, in_dat):
    n = shape(in_dat)[1]
    x = mat(ones((1, n+1)))
    x[:, 1:n+1] = in_dat
    return float(x * model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree, in_data)

    if in_data[tree["sp_ind"]] > tree["sp_val"]:
        if is_tree(tree["left"]):
            return tree_fore_cast(tree["left"], in_data, model_eval)
        else:
            return model_eval(tree["left"], in_data)
    else:
        if is_tree(tree["right"]):
            return tree_fore_cast(tree["right"], in_data, model_eval)
        else:
            return model_eval(tree["right"], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    m = len(test_data)
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, mat(test_data[i]), model_eval)
    return y_hat


if __name__ == "__main__":
    # 1. 测试
    # 1.1 回归树
    # my_dat = load_dataset('resource/ex2.txt')
    # my_mat = mat(my_dat)
    # my_tree = create_tree(my_mat, ops=(0, 1))
    #
    # my_dat_test = load_dataset('resource/ex2test.txt')
    # my_mat_test = mat(my_dat_test)
    # result = prune(my_tree, my_mat_test)
    # print(result)

    # 1.2 模型树
    # my_dat2 = load_dataset('resource/exp2.txt')
    # my_mat2 = mat(my_dat2)
    # my_tree2 = create_tree(my_mat2, model_leaf, model_err, (1, 10))
    # print(my_tree2)

    # 2. 骑自行车的速度和人的智商之间的关系
    # 2.1 回归树
    train_mat = mat(load_dataset('resource/bikeSpeedVsIq_train.txt'))
    test_mat = mat(load_dataset('resource/bikeSpeedVsIq_test.txt'))
    my_tree = create_tree(train_mat, ops=(1, 20))
    y_hat = create_fore_cast(my_tree, test_mat[:, 0])
    print(corrcoef(y_hat, test_mat[:, 1], rowvar=0)[0, 1])

    # 2.2 模型树
    my_tree2 = create_tree(train_mat, model_leaf, model_err, ops=(1, 20))
    y_hat2 = create_fore_cast(my_tree2, test_mat[:, 0], model_tree_eval)
    print(corrcoef(y_hat2, test_mat[:, 1], rowvar=0)[0, 1])

    # 3.3 标准线性回归
    ws, x, y = linear_solve(train_mat)
    y_hat3 = y_hat2.copy()
    for i in range(shape(test_mat)[0]):
        y_hat3[i] = test_mat[i, 0] * ws[1, 0] + ws[0, 0]
    print(corrcoef(y_hat3, test_mat[:, 1], rowvar=0)[0, 1])
