from numpy import *


def load_ex_data():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def load_ex_data2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 欧式距离
def eclud_sim(ina, inb):
    return 1.0/(1.0 + linalg.norm(ina - inb))


# 皮尔逊相关系数（ Pearson correlation）
def pears_sim(ina, inb):
    if len(ina) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(ina, inb, rowvar=0)[0][1]


# 余弦相似度
def cos_sim(ina, inb):
    num = float(ina.T * inb)
    denom = linalg.norm(ina) * linalg.norm(inb)
    return 0.5 + 0.5 * (num / denom)


def stand_est(data_mat, user, sim_meas, item):
    n = shape(data_mat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0

    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0:
            continue
        # 寻找两个用户都评级的物品
        over_lap = nonzero(logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]
        if len(over_lap) == 0:
            similarity = 0
        else:
            similarity = sim_meas(data_mat[over_lap, item], data_mat[over_lap, j])
        print("the {} and {} similarity is: {}".format(item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating

    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def recommend(data_mat, user, n=3, sim_meas=cos_sim, est_method=stand_est):
    # 寻找未评级的物品
    unrated_items = nonzero(data_mat[user, :].A == 0)[1]
    if len(unrated_items) == 0:
        return "you rated everthing"

    item_scores = []
    for item in unrated_items:
        estimated_score = est_method(data_mat, user, sim_meas, item)
        item_scores.append((item , estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[: n]


def svd_est(data_mat, user, sim_meas, item):
    n = shape(data_mat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    u, sigma, vt = linalg.svd(data_mat)
    # 建立对角矩阵
    sig4 = mat(eye(4) * sigma[:4])
    # 构建转换后的物品
    x_formed_items = data_mat.T * u[:, :4] * sig4.I

    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = sim_meas(x_formed_items[item, :].T, x_formed_items[j, :].T)
        print("the {} and {} similarity is: {}".format(item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating

    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def print_mat(in_mat, thresh=0.8):
    for i in range(32):
        line = []
        for k in range(32):
            if float(in_mat[i, k]) > thresh:
                line.append('1')
            else:
                line.append('0')
        print(','.join(line))


def img_compress(num_sv=3, thresh=0.8):
    myl = []
    with open("resource/0_5.txt", 'r') as fr:
        for line in fr.readlines():
            new_row = []
            for i in range(32):
                new_row.append(int(line[i]))
            myl.append(new_row)

    my_mat = mat(myl)
    print("****** original matrix ******")
    print_mat(my_mat, thresh)
    u, sigma, vt = linalg.svd(my_mat)
    sig_recon = mat(zeros((num_sv, num_sv)))
    for k in range(num_sv):
        sig_recon[k, k] = sigma[k]
    recon_mat = u[:, :num_sv] * sig_recon * vt[:num_sv, :]
    print("****** reconstructed matrix using {} singular values ******".format(num_sv))
    print_mat(recon_mat, thresh)


if __name__ == "__main__":
    # 1. 测试：
    # 1.1 小数据
    # u, sigma, vt = linalg.svd([[1, 1], [7, 7]])
    # print("u: {}".format(u))
    # print("sigma: {}".format(sigma))
    # print("vt: {}".format(vt))

    # 1.2 大一些的数据集
    # data = load_ex_data()
    # u, sigma, vt = linalg.svd(data)
    # print("sigma: {}".format(sigma))

    # 2. 基于协同过滤的推荐引擎
    # my_mat = mat(load_ex_data())
    # print(eclud_sim(my_mat[:, 0], my_mat[:, 4]))
    # print(eclud_sim(my_mat[:, 0], my_mat[:, 0]))
    # print(cos_sim(my_mat[:, 0], my_mat[:, 4]))
    # print(cos_sim(my_mat[:, 0], my_mat[:, 0]))
    # print(pears_sim(my_mat[:, 0], my_mat[:, 4]))
    # print(pears_sim(my_mat[:, 0], my_mat[:, 0]))

    # 3. 示例：餐馆菜肴推荐系统
    # 3.1 调整原始矩阵，验证
    # my_mat = mat(load_ex_data())
    # my_mat[0, 1] = my_mat[0, 0] = my_mat[1, 0] = my_mat[2, 0] = 4
    # my_mat[3, 3] = 2
    # print(my_mat)
    # print(recommend(my_mat, 2))
    # print(recommend(my_mat, 2, sim_meas=eclud_sim))
    # print(recommend(my_mat, 2, sim_meas=pears_sim))

    # 3.2 用稍大的数据集测试
    # my_mat = mat(load_ex_data2())
    # u, sigma, vt = linalg.svd(my_mat)
    # print("sigma: {}".format(sigma))
    # print(recommend(my_mat, 1, est_method=svd_est))
    # print(recommend(my_mat, 1, est_method=svd_est, sim_meas=pears_sim))

    # 4. 示例：图像压缩
    img_compress(2)

