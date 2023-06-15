def load_dataset():
    return [[1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]]


def create_c1(dataset):
    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset, c1))


def scan_d(d, ck, min_support):
    ss_cnt = {}
    for tid in d:
        for can in ck:
            if can.issubset(tid):
                ss_cnt[can] = ss_cnt.get(can, 0) + 1

    num_items = float(len(d))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        # 计算所有项的支持度
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(lk, k):
    ret_list = []
    len_lk = len(lk)

    for i in range(len_lk):
        for j in range(i+1, len_lk):
            l1 = list(lk[i])[: k-2]
            l2 = list(lk[j])[: k-2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                ret_list.append(lk[i] | lk[j])
    return ret_list


def apriori(dataset, min_support=0.5):
    c1 = create_c1(dataset)
    d = list(map(set, dataset))
    l1, support_data = scan_d(d, c1, min_support)
    l = [l1]
    k = 2

    while len(l[k-2]) > 0:
        ck = apriori_gen(l[k-2], k)
        lk, sup_k = scan_d(d, ck, min_support)
        support_data.update(sup_k)
        l.append(lk)
        k += 1
    return l, support_data


def generate_rules(l, support_data, min_conf=0.7):
    big_rule_list = []
    for i in range(1, len(l)):
        for freq_set in l[i]:
            hl = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_conseq(freq_set, hl, support_data, big_rule_list, min_conf)
            else:
                calc_conf(freq_set, hl, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, h, support_data, brl, min_conf=0.7):
    pruned_h = []
    for conseq in h:
        conf = support_data[freq_set] / support_data[freq_set - conseq]
        if conf >= min_conf:
            print("{} --> {}, conf: {}".format(freq_set - conseq, conseq, conf))
            brl.append((freq_set - conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_conseq(freq_set, h, support_data, brl, min_conf=0.7):
    m = len(h[0])
    if len(freq_set) > m + 1:
        hmp1 = apriori_gen(h, m + 1)
        hmp1 = calc_conf(freq_set, hmp1, support_data, brl, min_conf)
        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)


if __name__ == "__main__":
    # 1. 测试
    # 1.1 找出频繁项集
    # dataet = load_dataset()
    # c1 = create_c1(dataet)
    # print("c1: {}".format(c1))
    #
    # d = list(map(set, dataet))
    # l1, supp_data0 = scan_d(d, c1, 0.5)
    # print("l1: {}".format(l1))
    #
    # l1, supp_data0 = apriori(dataet, min_support=0.5)
    # print("l1: {}".format(l1))

    # 1.2 挖掘关联规则
    # rules = generate_rules(l1, supp_data0, min_conf=0.7)
    # print("rules: {}".format(rules))

    # 2. 示例：发现毒蘑菇的相似特征
    mush_dataset = [line.split() for line in open("resource/mushroom.dat").readlines()]
    l, supp_data = apriori(mush_dataset, min_support=0.3)
    for item in l[3]:
        if item.intersection('2'):
            print(item)

