from nltk import word_tokenize, sent_tokenize, RegexpTokenizer
from collections import Counter
punctuators_list = (',', '.', '!', '?', '"', '-', "'", ';', ':', '(', ')')
words_counter = Counter()

def is_number(word):
    try:
        float(word)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(word)
        return True
    except (TypeError, ValueError):
        pass

    return False


def replace_number_tokens(tokens):
    return list(map(lambda w: "N" if is_number(w) else w, tokens))


def remove_punctuators_from_tokens(tokens):
    return list(filter(lambda w: w not in punctuators_list, tokens))


def segment_files(filenames):
    segmented_text = []
    for file in filenames:
        with open(file) as f:
                f_text = f.read()
                sentences = sent_tokenize(f_text.lower())
                file_key = file+".out"
                tokenized_file_dict = {file_key:[]}
                for sent in sentences:
                    words = remove_punctuators_from_tokens(replace_number_tokens(word_tokenize(sent)))
                    tokenized_sent = []

                    for word in words:
                        if word is not "N":
                            words_counter[word] += 1
                            tokenized_sent.append(word)
                    tokenized_file_dict[file_key].append(tokenized_sent)
                segmented_text.append(tokenized_file_dict)

    return segmented_text


def get_most_common_words(top):
    return list(map(lambda  t: t[0],words_counter.most_common(top)))


def replace_uncommon_tokens(corpus, top):
    most_common_words = get_most_common_words(top)
    print(most_common_words)
    for file_dict in corpus:
        for file_id in file_dict.keys():
            for sent in file_dict[file_id]:
                for i in range(len(sent)):
                    current_word = sent[i]
                    if current_word not in most_common_words:
                        sent[i] = "< u n k >"


def write_sentences_to_file(file_corpus):
    for file_dict in file_corpus:
        for file_name in file_dict.keys():
            with open(file_name,'w+') as f:
                for sent_list in file_dict.values():
                    for sent in sent_list:
                        for word in sent:
                         f.write(str(word)+" ")
                        f.write("\n")


def ptb_preprocess(filenames, top=2):
    segmented_corpus = segment_files(filenames)
    replace_uncommon_tokens(segmented_corpus, top)
    write_sentences_to_file(segmented_corpus)
if __name__ == "__main__":
    ptb_preprocess(["D:\\test1.txt", "D:\\test2.txt"])
