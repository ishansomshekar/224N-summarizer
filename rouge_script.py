import rougescore as rs
import glob
import csv
import nltk
import os.path

model_name = "pointer_network_CE"


def gen_rouge(file_name):
    with open(model_name + 'csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rouge_l", "bleu_score"])
        count = 0

        with open(file_name, 'r') as f:
            while True:
                gen = f.readline()
                gold = f.readline()
                if not gold: 
                    break

                if gen != '' or gold != '':
                    rouge_l_score = rs.rouge_l(gen, [gold], 0.5)
                    # rouge_1_score = rs.rouge_n(gen, [gold], 1, 0.5)
                    # rouge_2_score = rs.rouge_n(gen, [gold], 2, 0.5)
                    # rouge_3_score = rs.rouge_n(gen, [gold], 3, 0.5)
                    bleu_score = nltk.translate.bleu_score.corpus_bleu([gen], [gold]) 
                writer.writerow([rouge_l_score, bleu_score])
                count += 1
                if count % 1000 == 0:
                    print count



def main():


if __name__ == '__main__':
    file_arg = sys.argv[]