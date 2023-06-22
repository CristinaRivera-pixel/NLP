import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator
####

#Utility functions
#Fill in the generator function get ngrams(n, text). The argument n is an int that
#tells you the size of the n-grams (you can assume n > 0), and the argument text is a list
#of strings (words) making up a sentence. The function should do the following:
#• Pad text with enough start tokens `<s>' so that you're able to make n-grams for
#the beginning of the sentence, plus a single end token `</s>', which we will need
#later in Part 3.
#• For each \real," non-start token, yield an n-gram tuple of the form (word, context),
#where word is a string and context is a tuple of the n􀀀1 preceding words/strings.

# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings

def appendstartendtokenstolist(text: List[str]):#my function i made but i didnt end up using
    sequence = list(text)

    sequence.append("</s>")
    sequence.insert(0,"<s>")

def getWordMakeTupes(text:List[str],wordIndex:int,bigramCount:int):#function i made didnt end up using

    singleList=[]
    newList=[]

    #singleList.append((text[wordIndex]))

    """for x in range(wordIndex-bigramCount,wordIndex):
        #print(text[x])
        singleList.append(text[x])"""

    for x in range(wordIndex-bigramCount,wordIndex):
        #print(text[x])
        singleList.append(text[x])

    newList.append(text[wordIndex])
    newList.append(tuple(singleList))

    return tuple(newList)


    return sequence


####

def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    # add start tokens (add n-1 start tokens)
    startTokenNums = n - 1

    # will insert n-1 start tokens
    for i in range(0, startTokenNums):
        text.insert(0, "<s>")

    text.append("</s>")  # adds a single end token

    # yielding n-gram tuple of the form (string, context),
    for i in range(startTokenNums, len(text)):
        # print(i)
        newList = []  # newlist stores all contexts
        for j in range(i - n + 1, i):
            # print(j)
            newList.append(text[j])

        contextTuple = tuple(newList)

        yield ((text[i], contextTuple, None, None))


# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    sentences = []

    wordList = []

    # Open the file at corpus path and load the text.

    f = open(corpus_path, "r")
    textFile = f.read()
    f.close()

    # Split the text into paragraphs. The corpus file has blank lines separating paragraphs.

    paragraphs_newLines = textFile.split("\n\n")

    # split paragraphs into sentences using sent tokens
    for x in range(len(paragraphs_newLines)):

        tokenized_pargraph = sent_tokenize(paragraphs_newLines[x])

        for y in range(len(tokenized_pargraph)):
            sentences.append(tokenized_pargraph[y])

    # turn those sentences into tokenized sentences
    for x in range(len(sentences)):
        tokenziedSent = word_tokenize(sentences[x])
        wordList.append(tokenziedSent)

    # print(wordList)
    return wordList



# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    n_gram_model = NGramLM(n)
    words = load_corpus(corpus_path)
    #print(words)

    for x in range(len(words)):
        n_gram_model.update(words[x])
        #print(words[x])

    return n_gram_model


# An n-gram language model
class NGramLM:

    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        for indv_ngrams in get_ngrams(self.n, text):
            # print(indv_ngrams[0:2])

            ngramKey = (indv_ngrams[0:2])
            # print(ngramKey)

            if (ngramKey in self.ngram_counts):
                self.ngram_counts[ngramKey] = self.ngram_counts[ngramKey] + 1
            else:
                self.ngram_counts[ngramKey] = 1

            self.vocabulary.add(indv_ngrams[0])
            contextKey = indv_ngrams[1]

            if (contextKey in self.context_counts):
                self.context_counts[contextKey] = self.context_counts[contextKey] + 1
            else:
                self.context_counts[contextKey] = 1

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta=.0) -> float:
        ngramCountGiven = 0

        if (context in self.context_counts):
            context_counts_given = self.context_counts[context]
        else:
            vocablength = len(self.vocabulary)
            probability = 1 / vocablength
            return probability

        if ((word, context) in self.ngram_counts):
            ngramCountGiven = self.ngram_counts[(word, context)]
            numera = ngramCountGiven + delta
            denomin = context_counts_given + delta * len(self.vocabulary)
            probability = numera / denomin

        else:
            numerator = (ngramCountGiven + delta)
            denom = (context_counts_given + delta * len(self.vocabulary))
            probability = numerator / denom

        return probability

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:

        prob = []

        total = 0

        for x in get_ngrams(self.n, sent):
            prob.append(self.get_ngram_prob(x[0], x[1], delta))

        logProb = []

        for num_val in prob:

            if (num_val == 0):
                newNum = -1 * math.inf
                logProb.append(newNum)
            else:
                newNum = math.log(num_val, 2)
                logProb.append(newNum)

        for ele in range(0, len(logProb)):
            total = total + logProb[ele]

        return total

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:

        num_counts = []

        for x in corpus:
            num_counts.append(len(x))

        # print(num_counts)
        Num = sum(num_counts)

        someString = ""

        for word in corpus:
            for ele in word:
                someString = someString + ele + " "


        log_prob = self.get_sent_log_prob(word_tokenize(someString))

        negEntropy = -1 * log_prob / (Num)

        perplexity = 2. ** negEntropy

        return perplexity

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:

        # print(random.random())

        # sort self.vocab

        r = random.random()

        self.vocabulary = sorted(self.vocabulary)

        zone = [0, 0]

        printStatement = " nothing was found"

        # generate a random number

        for vocab in self.vocabulary:
            probability = self.get_ngram_prob(vocab, context, delta)

            zone[0] = zone[1]
            zone[1] = zone[1] + probability

            if (r <= zone[1] and r >= zone[0]):
                return vocab

        return printStatement

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        context_list = []

        sentence_generated = ""

        for k in range(self.n - 1):
            context_list.append('<s>')

        if (max_length == 0):
            return sentence_generated
        for k in range(max_length):

            contextword = tuple(context_list)

            word = self.generate_random_word(contextword, delta)

            del context_list[0]

            context_list.append(word)

            if (word == '</s>'):
                sentence_generated = sentence_generated + word
                break
            sentence_generated = sentence_generated + word + " "

        return sentence_generated


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    NGramLM.update(s1)

    print(trigram_lm.get_sent_log_prob(word_tokenize(s1)))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2)))
    print(trigram_lm.generate_random_word(('I', 'am'), 2))
    print(trigram_lm.generate_random_text(5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
