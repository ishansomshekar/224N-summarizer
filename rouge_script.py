import rougescore as rs
import glob
import csv

import os.path
import sys

model_name = "pointer_network_CE"


def gen_rouge(file_name):
    print file_name
    with open('ROUGE_scores_' + file_name + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rouge_l"])
        count = 0
        rouge_scores = []
        with open(file_name + '.txt', 'r') as f:
            while True:
                gen = f.readline()
                gold = f.readline()
                f.readline()
                if not gold: 
                    break

                if gen != '' or gold != '':
                    rouge_l_score = rs.rouge_l(gen, [gold], 0.5)
                    # rouge_1_score = rs.rouge_n(gen, [gold], 1, 0.5)
                    # rouge_2_score = rs.rouge_n(gen, [gold], 2, 0.5)
                    # rouge_3_score = rs.rouge_n(gen, [gold], 3, 0.5)
                    # bleu_score = nltk.translate.bleu_score.corpus_bleu([gen], [gold]) 
                writer.writerow([rouge_l_score])
                rouge_scores.append(rouge_l_score)
                count += 1
                if count % 1000 == 0:
                    print count
            avg_r = float(sum(rouge_scores))/len(rouge_scores)
            print avg_r
            writer.writerow([avg_r])                    


def main(argv):
    gen_rouge(argv[0])


if __name__ == '__main__':
    main(sys.argv[1:])


