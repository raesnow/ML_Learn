import copy
import math
import operator
import pickle

from tree_plotter import create_plot


def calc_shannon_ent(dataset):
    """
    根据最后一列label值，计算香农熵

    :param dataset:
    :return:
    """
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_babel = feat_vec[-1]
        label_counts[current_babel] = label_counts.get(current_babel, 0) + 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ["no_surfacing", "flippers"]
    return dataset, labels


def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 创建唯一的分类标签索引
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        # 计算每种划分方式的信息熵
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        # 计算最好的信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, input_labels):
    labels = copy.deepcopy(input_labels)
    class_list = [example[-1] for example in dataset]
    # 类别完全相同，则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 遍历完所有特征时，返回出现次数最多的
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    # 得到列表包含的所有属性值
    feat_values = [example[best_feat] for example in dataset]
    unique_values = set(feat_values)
    # 递归，构建决策树
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    # 将标签字符串转换为索引
    feat_index = feat_labels.index(first_str)
    class_label = None
    for key in second_dict:
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(input_tree, f)


def grab_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # 1. 测试验证
    # dataset, labels = create_dataset()
    # my_tree = create_tree(dataset, labels)
    # print(classify(my_tree, labels, [1, 0]))
    # print(classify(my_tree, labels, [1, 1]))
    #
    # filename = 'classifier_storage.txt'
    # store_tree(my_tree, filename)
    # print(grab_tree(filename))

    # 2. 预测隐形眼镜类型
    lenses = []
    with open('lenses.txt', 'r') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
    lenses_tree = create_tree(lenses, lenses_labels)
    create_plot(lenses_tree)
