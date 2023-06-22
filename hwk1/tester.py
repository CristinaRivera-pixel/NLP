# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator

sentence = "I really like python, it's pretty awesome.".split()
#print(sentence)

#print(sentence) = ['I', 'really', 'like', 'python,', "it's", 'pretty', 'awesome.']
#how to access sentence
"""
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    print(n+text)
    pass
"""

"""def get_ngrams(n: int, text: List[str])-> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    if n <= 2:
        sequence = ['<s>'] + list(text) + ['</s>']
    else:
        sequence = ['<s>'] * (n - 1) + list(text) + ['</s>']

    print(sequence)
    ans = []
    for i in range(n - 1, len(sequence)):#lets say n=3 then n-1=2 len of sequence will be 7 so 2-7
        temp = []
        for j in range(i - n, i):#starts at 0,
            print("what is this "+sequence[j+1])
            temp.append(sequence[j + 1])
            print(ans)
        ans.append(tuple(temp))
    return ans

    pass"""


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
        get_ngrams(self.n, text)

        pass

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta=.0) -> float:
        pass

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        pass

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        pass

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        pass

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        pass


"""def appendstartendtokenstolist(text: List[str]):
    sequence = text
    sequence.append("</s>")

    sequence.insert(0,"<s>")#put in loop to get 
    return sequence"""


def getWordMakeTupes(text:List[str],wordIndex:int,bigramCount:int):

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



def make_tuples(tupleSize: int ,text:List[str]):

    #we want to start at 2 for the for loop
    #we want to end at len(text)+1
    newTuple=(0)
    returnTuple=()


    for x in range(2,len(text)-1):
       # print(text[x])
       firstTuple=(text[0])
        #for y in range(tupleSize):


    return newTuple








    #print(text)




def get_ngrams(n: int, text: List[str])-> Generator[Tuple[str, Tuple[str, ...]], None, None]:

    for x in range(n):
        text.insert(0,"<s>")

    text.append("</s>")


    #yield((text[i], tuple(context), None, None))
    """for x in range(len(text)): # x is the word in a list,
        print(x)"""


    for x in range(1,len(text)):
        newList.append(getWordMakeTupes(text,x,n))

    return newList





def read_text(textfile: str):

    with open(textfile) as f:
        contents = f.read()

    contents=contents.splitlines()





    return contents







def load_corpus(corpus_path: str) -> List[List[str]]:
    split_text=corpus_path.split()

    print(split_text)




    pass

"""f = open('warpeace.txt')
newlist=f.readlines()  # Returns a list object
f.close()"""

i=1
List=["hi","my"]

#print(get_ngrams(4,sentence))
#print(appendstartendtokenstolist(sentence))
#get_ngrams(2,sentence)

print(get_ngrams(2,sentence))



#print(read_text("shakespeare.txt"))