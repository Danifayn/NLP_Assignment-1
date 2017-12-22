import math
from collections import *
from nltk import probability
from  nltk.probability import FreqDist, ConditionalFreqDist
from  nltk.tokenize import  word_tokenize
import gathering_Basic_Statistics
import  matplotlib.pyplot as plt
# from  nltk import  *
stat_dict = {}


def normalize(counter):
    s = float(sum(counter.values()))
    return [(c, cnt / s, s) for c, cnt in counter.items()]


def lidstone(counter, diff_gamma=False):
    freq_dist = FreqDist(wrd for wrd in counter.keys())
    lids = []
    if diff_gamma:
        for i in range(1, 11):
            lid = probability.LidstoneProbDist(freq_dist, float(i) * 0.1)
            lids.append(lid)
    elif not diff_gamma:
        lid = probability.LidstoneProbDist(freq_dist, 1)
        lids.append(lid)

    return lids


def laplace(counter):
    freq_dist = FreqDist(wrd for wrd in counter.keys())
    lap = probability.LaplaceProbDist(freq_dist)
    return [lap]


def log(number):
    _NINF = float('-1e300')
    return math.log(number, 2) if number != 0 else _NINF


def format_dataset(dataset):
    return 1


def entropy(lm, test_file, order=2, diff_gamma=False):
    global stat_dict
    perplexity_list = []
    if diff_gamma:
        gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        gamma_list = [1]
    p_total_value = []
    for i in range(len(gamma_list)):
        p_total_value.append(0)

    v_size = stat_dict['Vocabulary size']
    # v_size = 10001
    with open(test_file) as tf:
        data = (tf.read()).split('\n')
        for sent in data:
            data[list(data).index(sent)] = ("<s> " * (order - 1)) + "<s> " + sent + " </s>"

        data = (" ".join(data)).split(" ")

        # the all test set as a sequence:
        print("LEN: ", len(data))
        for i in range(len(data)):
            h_key = ""
            for w in data[i:i + order]:
                h_key += w + " "

            h_key = h_key[:len(h_key) - 1]
            p_values = []
            if "</s>" not in h_key:
                # check if history is known

                if lm.get(h_key) is not None:
                   # print("history exists in model")
                    # check if history:data(i) are in train
                    for lid in lm[h_key]:
                        p_value = log(lid.prob(data[i]))
                        p_values.append(p_value)

                else:
                   # print("history doesn't exist in model")
                    for x in range(len(gamma_list)):
                        p_value = (log(1 / v_size))
                        p_values.append(p_value)

            for j in range(len(p_values)):
                p_total_value[j] = p_total_value[j] + p_values[j]

        best_gamma_value = 0
        min_h_value = -p_total_value[0] / (float(len(data)))
        min_perp_value = pow(2, min_h_value)
        for i in range(len(p_total_value)):
            h_value = -p_total_value[i] / (float(len(data)))
            perplexity_value = pow(2, h_value)
            if h_value <= min_h_value:
                min_h_value = h_value
            if perplexity_value <= min_perp_value:
                min_perp_value = perplexity_value
                best_gamma_value = gamma_list[i]
            print("ENTROPY: ", h_value, " For gamma=",gamma_list[i], " For n-gram size=", order + 1)
            print("PERPLEXITY: ", perplexity_value)
            perplexity_list.append(perplexity_value)
        print("MIN ENTROPY: ", min_h_value)
        print("MIN PERPLEXITY: ", min_perp_value)

        perplexity_list.insert(0,min_perp_value)
        print("MIN PERPLEXITY IN LIST : ", perplexity_list[0])
  #      perplexity_gamma_graph(perplexity_list, gamma_list)
        return [min_h_value, perplexity_list, best_gamma_value]


def perplexity(lm, test_file, order=2, diff_gamma=False):
    return math.pow(2, entropy(lm, test_file, order, diff_gamma)[0])


def train_word_lm(dataset, estimator, n=2, diff_gamma=False):
    global stat_dict
    stat_dict = gathering_Basic_Statistics.get_statistics(dataset)
    for file_name in dataset:
        with open(file_name) as f:
            data = (f.read()).split('\n')

            for sent in data:

                data[list(data).index(sent)] = ("<s> " * (n-1)) + "<s> " + sent[1:len(sent)-1] + " </s>"

            lm = defaultdict(Counter)
            data = (" ".join(data)).split(" ")

            for i in range(len(data) - n):
                history, word = "", data[i + n]

                for w in data[i:i + n - 1]:
                    history += w + " "
                history += data[i + n - 1]
                if "</s>" not in history:
                    lm[history][word] += 1
            if diff_gamma:
                outlm = {hist: lidstone(words, True) for hist, words in lm.items()}
            else:
                outlm = {hist: estimator(words) for hist, words in lm.items()}

            print("Trained model")
            return outlm


def test_lm_diff_gamma(train_files, test_file, n=1):
    lm = train_word_lm(train_files, lidstone, n, True)
    entropy(lm, test_file, n, True)


def test_lm(train_files, test_file, n=1, inc_flag=False, n_inc=9, estimator=lidstone, diff_gamma=False):
    global stat_dict
    stat_dict = gathering_Basic_Statistics.get_statistics(train_files)
   # if diff_gamma:
       # test_lm_diff_gamma(train_files, test_file,n)
   # else:
    if inc_flag:
        if not diff_gamma:
            perp_list = []
            n_inc_list = []
            for i in range(n+1, n_inc + 2):
                n_inc_list.append(i)
            for i in range(n, n_inc + 1):
                lm = train_word_lm(train_files, estimator, i)
                perp_list.append(entropy(lm, test_file, i)[1][0])
            n_inc_perplexity_graph(n_inc_list, perp_list)
        else:
            res = 1
            perp_list = []
            n_inc_list = []
            for i in range(n + 1, n_inc + 2):
                n_inc_list.append(i)
            for i in range(n, n_inc + 1):
                lm = train_word_lm(train_files, lidstone, i, True)
                res = entropy(lm, test_file, i, True)
                perp_list.append(res[1][0])
            n_inc_perplexity_graph(n_inc_list, perp_list, res[2])

    elif not inc_flag:
        lm = train_word_lm(train_files, estimator, n)
        entropy(lm, test_file, n)


def n_inc_perplexity_graph(n_inc_list, p_list, gamma_value):
    plt.plot(n_inc_list, p_list)
    plt.xlabel("n-gram size")
    plt.ylabel("Perplexity for gamma value = "+str(gamma_value))
    plt.show()


def perplexity_gamma_graph(p_list, g_list):
    plt.plot(g_list,p_list)
    plt.xlabel('gamma value')
    plt.ylabel('Perplexity')
    plt.show()

if __name__ == "__main__":
    dataset_files_names = [r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.train.txt']
    valid_set = r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.valid.txt'
    test_set = r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.test.txt'
    # Best possible n order : 3-gram ,gamma : 1 perplexity on test is : ~95
    test_lm(dataset_files_names, test_set, 2, True, 4, diff_gamma=True)

