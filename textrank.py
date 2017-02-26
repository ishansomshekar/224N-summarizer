"""
From this paper:
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

External dependencies: nltk, numpy, networkx

Based on https://gist.github.com/voidfiles/1646117
         https://github.com/davidadamojr/TextRank
"""

from os import listdir
from os.path import isfile, join, getsize
from collections import defaultdict
from collections import Counter
from os import getcwd

# import nltk
import networkx as nx
# import re
from summa import summarizer
from summa import keywords




"""

# apply syntactic filters based on POS tags
def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    return [item for item in tagged if item[1] in tags]


def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in [x for x in iterable if x not in seen]:
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def lDistance(firstString, secondString):
    Function to find the Levenshtein distance between two words/sentences -
    gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python
    
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]


def buildGraph(nodes):
    
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = lDistance(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr


def extractKeyphrases(text):
    # tokenize the text using nltk
    wordTokens = nltk.word_tokenize(text)

    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(wordTokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = buildGraph(word_set_list)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    aThird = len(word_set_list) // 3
    keyphrases = keyphrases[0:aThird + 1]

    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modifiedKeyphrases = set([])
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    dealtWith = set([])
    i = 0
    j = 1
    while j < len(textlist):
        firstWord = textlist[i]
        secondWord = textlist[j]
        if firstWord in keyphrases and secondWord in keyphrases:
            keyphrase = firstWord + ' ' + secondWord
            modifiedKeyphrases.add(keyphrase)
            dealtWith.add(firstWord)
            dealtWith.add(secondWord)
        else:
            if firstWord in keyphrases and firstWord not in dealtWith:
                modifiedKeyphrases.add(firstWord)

            # if this is the last word in the text, and it is a keyword, it
            # definitely has no chance of being a keyphrase at this point
            if j == len(textlist) - 1 and secondWord in keyphrases and \
                    secondWord not in dealtWith:
                modifiedKeyphrases.add(secondWord)

        i = i + 1
        j = j + 1

    return modifiedKeyphrases


def extractSentences(text):
    # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # sentenceTokens = sent_detector.tokenize(text.strip())
    sentenceTokens = re.split(r'[.;]', text)

    graph = buildGraph(sentenceTokens)
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    # return a 100 word summary
    summary = ' '.join(sentences)
    summaryWords = summary.split()
    summaryWords = summaryWords[0:101]
    summary = ' '.join(summaryWords)

    return summary


def writeFiles(summary, keyphrases, fileName):
    "outputs the keyphrases and summaries to appropriate files"
    print("Generating output to " + 'keywords/' + fileName)
    keyphraseFile = io.open('keywords/' + fileName, 'w')
    for keyphrase in keyphrases:
        keyphraseFile.write(keyphrase + '\n')
    keyphraseFile.close()

    print("Generating output to " + 'summaries/' + fileName)
    summaryFile = io.open('summaries/' + fileName, 'w')
    summaryFile.write(summary)
    summaryFile.close()

    print("-")



def summarize_all():
    # retrieve each of the articles
    articles = os.listdir("articles")
    for article in articles:
        print('Reading articles/' + article)
        articleFile = io.open('articles/' + article, 'r')
        text = articleFile.read()
        keyphrases = extractKeyphrases(text)
        summary = extractSentences(text)
        writeFiles(summary, keyphrases, article)

with open("bills/sconres_109_5_is.txt") as fin:
    text = fin.read()
    summary = extractSentences(text)

    summaryFile = open('bills/summary_sconres_109_5_is.txt', 'w')#io.open('bills/bill_summary', 'w')    
    summaryFile.write(summary)
    summaryFile.write("MIT implementation:")    
    summaryFile.write ('\n\n');
    summaryFile.write(summarizer.summarize(text, ratio=0.2))
    summaryFile.write(keywords.keywords(text, words= 10))
    # print(summary)
"""

def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]

def get_bill_ID(file):
    names = file.split('/')
    ID = names[len(names)-1].split('.')[0]
    return ID

def generate_summaries(filePath, newFilePath):
    fileNames = return_files(filePath)

    count = 0
    for file in fileNames:
        if count % 100 == 0:
            print "finished %d" % count
        text = ''
        with open(file, 'r') as f:
            text = f.read()


        f.close()

        newFileName = get_bill_ID(file) + '_generated'
        # print "#### text"
        # print text
        # print "####\n"
        summary = summarizer.summarize(text, words = 50)
        
        # print "#### summary"
        # print summary
        # print "####\n"

        newFile = open(newFilePath + newFileName + '.txt', 'w')
        newFile.write(summary)
        newFile.close()

        count += 1







def main(args):
    curDir = getcwd()
    txt_data_path = curDir + '/txt_files/processed/test/'
    new_files_path = curDir + '/generated_summaries/test/'
    xml_data_path = curDir + '/xml_files/processed/'
    generate_summaries(txt_data_path, new_files_path)



if __name__=='__main__':
    import sys
    main(sys.argv[1:])

# if __name__ == '__main__':
#     cli()
