import rougescore as rs
import glob
import csv
import nltk

path_to_gold = 'gold_summaries/'
path_to_ours = 'our_summaries/*.txt'

with open('ROUGE_SCORES.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["bill", "rouge_l", "rouge_1", "rouge_2", "rouge_3", "bleu_score"])

    for filename in glob.glob(path_to_ours):
        our_summary = open(filename, 'r').read()
        sentence_suffix = "_our_summary.txt"
        sentence_prefix = "our_summaries"
        file_body = filename[len(sentence_prefix) + 1:-len(sentence_suffix)]
        file_name = file_body + "_gold_summary"
        gold_summary = open(path_to_gold + file_name).read()

        rouge_l_score = rs.rouge_l(our_summary, [gold_summary], 0.5)
        rouge_1_score = rs.rouge_n(our_summary, [gold_summary], 1, 0.5)
        rouge_2_score = rs.rouge_n(our_summary, [gold_summary], 2, 0.5)
        rouge_3_score = rs.rouge_n(our_summary, [gold_summary], 3, 0.5)
        
        bleu_score = nltk.translate.bleu_score.corpus_bleu([our_summary], [gold_summary])
        writer.writerow([file_body, rouge_l_score, rouge_1_score, rouge_2_score, rouge_3_score, bleu_score])
