class TreeNode:
    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def disp(self, ind=1):
        print("{}{} {}".format(' '*ind, self.name, self.count))
        for child in self.children.values():
            child.disp(ind+1)


def create_tree(dataset, min_sup=1):
    header_table = {}
    for trans in dataset:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + dataset[trans]
    # 移除不满足最小支持度的元素项
    new_header_table = {}
    for k in header_table:
        if header_table[k] >= min_sup:
            new_header_table[k] = header_table[k]
    header_table = new_header_table

    freq_item_set = set(header_table.keys())
    # 如果没有元素项满足要求，则退出
    if len(freq_item_set) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]

    ret_tree = TreeNode("Null Set", 1, None)
    for tran_set, count in dataset.items():
        local_d = {}
        for item in tran_set:
            if item in freq_item_set:
                local_d[item] = header_table[item][0]
        if len(local_d) > 0:
            # 根据全局频率对每个事务中的元素进行排序
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: p[1], reverse=True)]
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = TreeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])

    if len(items) > 1:
        update_tree(items[1:], in_tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


def load_simp_dat():
    simp_dat = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simp_dat


def create_init_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


def ascend_tree(leaf_node, prefix_path):
    # 迭代上溯整棵树
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    cond_pats = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats


def mine_tree(in_tree, header_table, min_sup, pre_fix, freq_item_list):
    # 从头指针表的底端开始
    big_l = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]

    for base_pat in big_l:
        new_freq_set = pre_fix.copy()
        new_freq_set.add(base_pat)
        freq_item_list.append(new_freq_set)
        cond_patt_bases = find_prefix_path(base_pat, header_table[base_pat][1])
        # 从条件模式基来构建条件FP树
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_sup)
        if my_head is not None:
            # 挖掘条件FP树
            print("conditional tree for: {}".format(new_freq_set))
            my_cond_tree.disp(1)
            mine_tree(my_cond_tree, my_head, min_sup, new_freq_set, freq_item_list)


if __name__ == "__main__":
    # 1. 测试
    # 1.1 树节点
    # root_node = TreeNode('pyramid', 9, None)
    # root_node.children['eye'] = TreeNode('eye', 13, None)
    # root_node.children['phoenix'] = TreeNode('phoenix', 3, None)
    # root_node.disp()

    # 1.2 FP_growth
    # simp_dat = load_simp_dat()
    # init_set = create_init_set(simp_dat)
    # my_fp_tree, my_header_tab = create_tree(init_set, 3)
    # my_fp_tree.disp()
    # print(my_header_tab)
    # print(find_prefix_path('x', my_header_tab['x'][1]))
    # print(find_prefix_path('r', my_header_tab['r'][1]))
    #
    # freq_items = []
    # mine_tree(my_fp_tree, my_header_tab, 3, set([]), freq_items)
    # print(freq_items)

    # 2. 从新闻网站点击流中挖掘
    parsed_dat = [line.split() for line in open("resource/kosarak.dat").readlines()]
    init_set = create_init_set(parsed_dat)
    my_fp_tree, my_header_tab = create_tree(init_set, 100000)
    my_freq_list = []
    mine_tree(my_fp_tree, my_header_tab, 100000, set([]), my_freq_list)
    print(my_freq_list)