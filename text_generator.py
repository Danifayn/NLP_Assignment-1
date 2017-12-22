from n_gram_model import *
globals()

def generate_word(lm, history, order):
    history = ' '.join(history)
    print(history)
    return lm[history][0].generate()

# def old_generate_word(lm, history, order):
#     print(history)
#     history = (' '.join(history[-order:]))
#     dist = lm[history]
#
#     while True:
#         x = random()
#         for w_c in dist.items():
#             x = x - w_c[1]
#             # print(dist)
#             print(x)
#             # print(w_c[0])
#             if x <= 0:
#                 return w_c[0]


def generate_text(lm, order, nsentences,nfiles):
    for z in range(nfiles):
        out_sent = []
        for y in range(nsentences):
            history = []
            out = []
            for x in range(order):
                history.append("<s>")
                out.append("<s>")
            while True:
                wrd = generate_word(lm, history, order)
                history.append(wrd)
                history = history[-order:]
                out.append(wrd)
                if wrd == '</s>':
                    out_sent.append((" ".join(out[order:-1]))+'\n')
                    break
        with open('generated_text_'+str(z+1)+'.txt', 'w+')as f:
            f.writelines(out_sent)


if __name__ == "__main__":
    lm1 = train_word_lm([r'D:\NLP\HW1_Stuff\simple-examples\data\ptb.train.txt'], lidstone, n=10)
    print(generate_text(lm1, 10, 5, 1))
    exit()
    entropy(lm1,'generated_text_1.txt', 2)
    entropy(lm1, 'generated_text_2.txt', 2)
    entropy(lm1, 'generated_text_3.txt', 2)
    entropy(lm1, 'generated_text_4.txt', 2)
    entropy(lm1, 'generated_text_5.txt', 2)