import pickle
import feedparser
from numpy import *
import re
import operator


def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性文字，0 代表正常言论
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(dataset):
    vocab_set = set()
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: {} is not in my vocabulary!".format(word))
    return return_vec


def bag_of_words2vec_mn(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nb_0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # 通过求对数可以避免下溢出或者浮点数舍入导致的错误
    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_of_posts, list_classes = load_dataset()
    my_vocab_list = create_vocab_list(list_of_posts)
    train_mat = []
    for post_in_doc in list_of_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = train_nb_0(train_mat, list_classes)

    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print(f"{test_entry} classified as: {classify_nb(this_doc, p0_v, p1_v, p_ab)}")

    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print(f"{test_entry} classified as: {classify_nb(this_doc, p0_v, p1_v, p_ab)}")


def text_parse(big_string):
    list_of_tokens = re.split(r'\W+', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []

    for i in range(1, 26):
        word_list = text_parse(open(f'resource/email/spam/{i}.txt').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        word_list = text_parse(open(f'resource/email/ham/{i}.txt').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)

    # 随机构建训练集
    training_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb_0(train_mat, train_classes)

    error_count = 0
    for doc_index in test_set:
        word_vector = set_of_words2vec(vocab_list, doc_list[doc_index])
        if classify_nb(word_vector, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print(f"the error rate is: {float(error_count) / len(test_set)}")


def calc_most_freq(vocab_list, full_text):
    """
    获取高频词汇
    :param vocab_list:
    :param full_text:
    :return:
    """
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:10]


def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    full_text = []

    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['title'])
        if 'summary' in feed1['entries'][i]:
            word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        word_list = text_parse(feed0['entries'][i]['title'])
        if 'summary' in feed0['entries'][i]:
            word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)
    # 剔除高频词
    top10words = calc_most_freq(vocab_list, full_text)
    for pair_w in top10words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])

    training_set = list(range(2 * min_len))
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words2vec_mn(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb_0(train_mat, train_classes)

    error_count = 0
    for doc_index in test_set:
        word_vector = bag_of_words2vec_mn(vocab_list, doc_list[doc_index])
        if classify_nb(word_vector, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print(f"the error rate is: {float(error_count) / len(test_set)}")
    return vocab_list, p0_v, p1_v


def get_top_words(feed1, feed0):
    vocab_list, p0v, p1v = local_words(feed1, feed0)
    top_feed1 = []
    top_feed0 = []
    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            top_feed0.append((vocab_list[i], p0v[i]))
        if p1v[i] > -6.0:
            top_feed1.append((vocab_list[i], p0v[i]))

    sorted_feed0 = sorted(top_feed0, key=lambda pair: pair[1], reverse=True)
    print("***** feed0 *****")
    for item in sorted_feed0[:10]:
        print(item[0])
    sorted_feed1 = sorted(top_feed1, key=lambda pair: pair[1], reverse=True)
    print("***** feed1 *****")
    for item in sorted_feed1[:10]:
        print(item[0])


if __name__ == "__main__":
    # 1. 测试示例
    # testing_nb()

    # 2. 过滤垃圾邮件
    # spam_test()

    # 3. 依据rss进行分类
    # CNN
    # ny = feedparser.parse('http://rss.cnn.com/rss/cnn_topstories.rss')
    # 南华早报（South China Morning Post，香港主要英文报纸）
    # ny = feedparser.parse('https://www.scmp.com/rss/91/feed')
    # with open('resource/feed_parse/scmp.rss', 'wb') as f:
    #     pickle.dump(ny, f)

    with open('resource/feed_parse/cnn.rss', 'rb') as f:
        cnn = pickle.load(f)
    with open('resource/feed_parse/scmp.rss', 'rb') as f:
        scmp = pickle.load(f)
    local_words(cnn, scmp)

    get_top_words(cnn, scmp)
