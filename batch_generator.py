from read_in_datafile import file_generator
from embedding_wrapper import EmbeddingWrapper
import numpy as np

def batch_generator(embedding_wrapper, bill_data_path, indices_data_path, sequences_data_path, key_words_datapath, batch_size, MAX_BILL_LENGTH):

    f_generator = file_generator(batch_size, bill_data_path, indices_data_path, sequences_data_path, key_words_datapath)

    #pad the bills and summaries
    print "now padding and encoding batches"
    padded_bills = []
    padded_start_indices = []
    padded_end_indices = []
    padded_masks = []
    padded_keywords = []
    for bill_batch, indices_batch, sequences, keywords in f_generator:
        # print "batch"
        # print bill_batch
        #print indices_batch
        for idx, bill in enumerate(bill_batch):
            start_index, end_index = indices_batch[idx]
            sequence_len = sequences[idx]
            keywords_batch = keywords[idx]
            bill_list = [embedding_wrapper.get_value(word) for word in bill.split()]
            padded_keyword = [embedding_wrapper.get_value(word) for word in keywords_batch]
            # padded_summary = [embedding_wrapper.get_value(word) for word in summary] d g
            mask = [True] * min(len(bill_list), MAX_BILL_LENGTH)
            padded_bill = bill_list[:MAX_BILL_LENGTH]
            # padded_summary = padded_summary[:MAX_SUMMARY_LENGTH]
            mask = mask[:MAX_BILL_LENGTH]

            for i in xrange(0, MAX_BILL_LENGTH - len(padded_bill)):
                padded_bill.append(embedding_wrapper.get_value(embedding_wrapper.pad))
                mask.append(False)

            for i in xrange(0, 5 - len(padded_keyword)):
                padded_keyword.append(embedding_wrapper.get_value(embedding_wrapper.pad))

            padded_masks.append(mask)
            padded_bills.append(padded_bill)

            start_index_one_hot = [0] * MAX_BILL_LENGTH
            if start_index >= MAX_BILL_LENGTH:
                start_index_one_hot[0] = 1
            else:
                start_index_one_hot[start_index] = 1

                            #now pad start_index_one_hot starting at sequence_len to be alternating 0 and 1 to mask loss
            if (len(start_index_one_hot) > len(bill_list)):
                val = 0
                for i in xrange(0, len(start_index_one_hot) - sequence_len):
                    start_index_one_hot[sequence_len + i] = val
                    val ^= 1

            #generate normal distribution
            # distrib = np.random.normal(0.6, 0.25, int(.25 * len(start_index_one_hot)))
            # distrib = [x for x in distrib if x < .95 and x > 0]
            # distrib = sorted(distrib, reverse = True)
            # #now, add around the two one hots
            # for idx, value in enumerate(distrib):
            #     idx += 1
            #     if (start_index - idx) > 0 and (start_index - idx) < len(start_index_one_hot):
            #         start_index_one_hot[start_index - idx] = value
            #     if (start_index + idx) < len(start_index_one_hot):
            #         start_index_one_hot[start_index + idx] = value

            # end_index_one_hot = [0] * MAX_BILL_LENGTH
            # if end_index >= MAX_BILL_LENGTH:
            #     end_index_one_hot[MAX_BILL_LENGTH - 1] = 1
            # else:
            #     end_index_one_hot[end_index] = 1

            # for idx, value in enumerate(distrib):
            #     idx += 1
            #     if (end_index - idx) > 0 and (end_index - idx) < len(end_index_one_hot):
            #         end_index_one_hot[end_index - idx] = value
            #     if (end_index + idx) < len(end_index_one_hot):
            #         end_index_one_hot[end_index + idx] = value

            # print end_index_one_hot

            # print start_index_one_hot
            # print

            # print "seq_len",sequence_len
            # print "number of words in bill: ", len(bill_list)
            # print "length of one hots:" ,len(start_index_one_hot)
            # print "one hots:", start_index_one_hot
            # print "different in padded and actual lengths", len(start_index_one_hot) - len(bill_list)
            # print

            padded_start_indices.append(start_index_one_hot)
            padded_end_indices.append(end_index_one_hot)
            padded_keywords.append(padded_keyword)

        yield padded_bills, padded_start_indices, padded_end_indices, padded_masks, sequences, padded_keywords
        padded_bills = []
        padded_start_indices = []
        padded_end_indices = []
        padded_masks = []
        padded_keywords = []

    #convert to integers
    print "finished inputting bills and summaries"
