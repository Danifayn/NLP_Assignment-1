from nltk import probability, Counter
from  nltk.probability import FreqDist, ConditionalFreqDist
from  nltk.tokenize import  word_tokenize
import  matplotlib.pyplot as plt
def laplace_stuff():
    sent = "am ate ate apple am x."
    sent_tokenized = word_tokenize(sent)
    freq_dist = FreqDist(word.lower() for word in word_tokenize(sent))
    print(freq_dist.items())
    lap = probability.LaplaceProbDist(freq_dist)
    print(lap.generate())
    print(lap.prob("am"))
    print("Finished freq dist, Starting Cond dist")
    # Cond Probabilty
    cond_dist = ConditionalFreqDist()
    context = None
    tokens = sent_tokenized
    # The type of the preceeding word
    for token in tokens:
        outcome = token
        cond_dist[context] = (outcome)
        context = token
    print(cond_dist["am"])
    print(cond_dist.items())


if __name__ == "__main__":
    laplace_stuff()