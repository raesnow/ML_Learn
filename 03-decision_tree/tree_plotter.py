import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt,
                             xycoords='axes fraction',
                             xytext=center_pt,
                             textcoords='axes fraction',
                             va='center',
                             ha='center',
                             bbox=node_type,
                             arrowprops=arrow_args)


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5/plot_tree.totalW  # 相对于中心点的偏移
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_trees = [
        {
            "no surfacing": {
                0: "no",
                1: {
                    "flippers": {
                        0: "no",
                        1: "yes"
                    }
                }
            }
        },
        {
            "no surfacing": {
                0: {
                    "head": {
                        0: "no",
                        1: "yes"
                    }
                },
                1: "no"
            }
        }]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    父子节点之间填充文本信息
    :param cntr_pt:
    :param parent_pt:
    :param txt_string:
    :return:
    """
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    # 计算宽与高
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW, plot_tree.yOff)

    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)

    second_dict = my_tree[first_str]
    plot_tree.yOff -= 1.0/plot_tree.totalD
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xOff += 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff += 1.0/plot_tree.totalD


if __name__ == "__main__":
    tree = retrieve_tree(0)
    create_plot(tree)

