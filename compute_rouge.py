import rougescore as rs
import glob
import csv
import nltk
import os.path

## download this github directory 
## https://github.com/bdusell/rougescore/blob/master/rougescore/rougescore.py



path_to_gold = 'gold_summaries/'
path_to_gen = 'generated_summaries/*.txt'

with open('ROUGE_SCORES.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["bill", "rouge_l", "rouge_1", "rouge_2", "rouge_3", "bleu_score"])
    count = 0
    for filename in glob.glob(path_to_gen):
        sentence_suffix = "_generated.txt"
        sentence_prefix = "generated_summaries"

        file_body = filename[len(sentence_prefix) + 1:-len(sentence_suffix)]

        gold_file_name = file_body + "_gold.txt"
        # print gold_file_name
        if os.path.isfile(path_to_gold + gold_file_name):
            if count % 100 == 0:
                print 'finished %d' % count
            # print filename
            gen_summary = open(filename, 'r').read()

            gold_summary = open(path_to_gold + gold_file_name).read()
            if gold_summary != '':
                rouge_l_score = rs.rouge_l(gen_summary, [gold_summary], 0.5)
                rouge_1_score = rs.rouge_n(gen_summary, [gold_summary], 1, 0.5)
                rouge_2_score = rs.rouge_n(gen_summary, [gold_summary], 2, 0.5)
                rouge_3_score = rs.rouge_n(gen_summary, [gold_summary], 3, 0.5)
            
                bleu_score = nltk.translate.bleu_score.corpus_bleu([gen_summary], [gold_summary])
                writer.writerow([file_body, rouge_l_score, rouge_1_score, rouge_2_score, rouge_3_score, bleu_score])
            count += 1


        