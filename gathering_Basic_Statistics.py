from collections import Counter
from nltk.util import ngrams
import  data_Exploration
import matplotlib.pyplot as plt
import matplotlib


def count_tokens():
    return 0


def count_chars(tokens):
    char_counter = 0
    for token in tokens:
        char_counter += len(token)
    return char_counter


def dataset_difference(d1, d2):
    with open(d1) as d1, open(d2) as d2:
        d1_text = d1.read()
        d2_text = d2.read()
        d1_tokens = str(d1_text).split(' ')
        d2_tokens = str(d2_text).split(' ')
        return len(set(d1_tokens) - set(d2_tokens))


def count_distinct_ngrams(tokens,size=4):
    char_ngrams_dict = {}
    for i in range(2, size + 1):
        char_ngrams_dict[i] = len(set(ngrams(tokens, i)))
    return char_ngrams_dict


def count_distinct_char_ngrams(file_text,size=7):
    char_ngrams_dict = {}
    for i in range(2,size+1):
        char_ngrams_dict[i] = len(set(ngrams(file_text,i)))
    return char_ngrams_dict


def get_statistics(dataset,top=4000):
    statistics_dict = {}
    token_counter = Counter()
    n_grams_words_dict = {}
    n_grams_chars_dict = {}
    top_count = 0
    for file_name in dataset:
        with open(file_name) as dataset:
            f_text = dataset.read()
            sent = f_text.split('\n')
            tokens = (" ".join(sent)).split(" ")

            print(tokens)
            num_of_tokens = len(tokens)
            if statistics_dict.get('Number of tokens') is None:
                statistics_dict['Number of tokens'] = 0
            statistics_dict['Number of tokens'] += num_of_tokens
            char_counter = count_chars(tokens)
            if statistics_dict.get('Number of characters') is None:
                statistics_dict['Number of characters'] = 0
            statistics_dict['Number of characters'] += char_counter
            vocabulary_tokens = set(tokens)
            statistics_dict['Vocabulary'] = vocabulary_tokens
            vocabulary_size = len(vocabulary_tokens)
            if statistics_dict.get('Vocabulary size') is None:
                statistics_dict['Vocabulary size'] = 0
            statistics_dict['Vocabulary size'] += vocabulary_size
            num_of_word_grams = count_distinct_ngrams(tokens)
            num_of_char_grams = count_distinct_char_ngrams(f_text)

            for key in num_of_word_grams.keys():
                if n_grams_words_dict.get(key) is None:
                    n_grams_words_dict[key] = 0
                n_grams_words_dict[key] += num_of_word_grams[key]

            for key in num_of_char_grams.keys():
                if n_grams_chars_dict.get(key) is None:
                    n_grams_chars_dict[key] = 0
                n_grams_chars_dict[key] += num_of_char_grams[key]

            for token in vocabulary_tokens:
                token_counter[token] += 1

    top_list = token_counter.most_common(top)
    for tuple in top_list:
        top_count += tuple[1]

    print("Printing top count: ", top_count)
    log_freq_rank(token_counter)
    statistics_dict['top-'+str(top)+' common tokens'] = top_count
    statistics_dict['token/type ratio'] = statistics_dict["Number of tokens"] / statistics_dict["Vocabulary size"]
    statistics_dict['Number of tokens in Dev but not in Train:'] = dataset_difference(dataset_files_names[1],
                                                                                      dataset_files_names[0])
    #print(n_grams_words_dict)
    #print(n_grams_chars_dict)
    # TODO The average number and standard deviation of characters per token
    return statistics_dict

def log_freq_rank(counter):
    print("graph stuff 1")
    plt.loglog([val for word , val in counter.most_common(4000)])
    print("graph stuff 2")
    plt.xlabel('rank')
    plt.ylabel('frequency')
    print("graph stuff 3")
  #  plt.show()
    print("Showed stuff")
    # ngrams = set(ngrams(word_tokenize("shalom alssag. sfsags"),2))
# print(ngrams)
dataset_files_names = [r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.train.txt',
                       r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.valid.txt',
                       r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.test.txt']

#num_of_tokenns = get_statistics(dataset_files_names)

#print(num_of_tokenns)
