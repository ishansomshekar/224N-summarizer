from read_in_datafile import file_generator
from embedding_wrapper import EmbeddingWrapper

def batch_generator(embedding_wrapper, bill_data_path, summary_data_path, batch_size, MAX_BILL_LENGTH, MAX_SUMMARY_LENGTH):

    f_generator = file_generator(batch_size, bill_data_path, summary_data_path)

    #pad the bills and summaries
    print "now padding and encoding batches"
    padded_batch = []
    for bill_batch, summary_batch in f_generator:
        # print "batch"
        print bill_batch
        #print summary_batch
        for idx, bill in enumerate(bill_batch):
            summary = summary_batch[idx]
            padded_bill = [embedding_wrapper.get_value(word) for word in bill]
            padded_summary = [embedding_wrapper.get_value(word) for word in summary]
            padded_bill = padded_bill[:MAX_BILL_LENGTH]
            padded_summary = padded_summary[:MAX_SUMMARY_LENGTH]

            for i in xrange(0, MAX_BILL_LENGTH - len(padded_bill)):
                padded_bill.append(embedding_wrapper.pad)

            for i in xrange(0, MAX_SUMMARY_LENGTH - len(padded_summary)):
                padded_summary.append(embedding_wrapper.pad)

            #now, change each word index into a one hot vector of size vocab size
            
            padded_bill_of_one_hots = [[0 for y in range(0, embedding_wrapper.num_tokens)] for x in range(0,MAX_SUMMARY_LENGTH)]
            for idx, id_word_representation in enumerate(padded_bill):
                padded_bill_of_one_hots[idx][id_word_representation] = 1

            padded_summary_of_one_hots = [[0 for y in range(0, embedding_wrapper.num_tokens)] for x in range(0,MAX_SUMMARY_LENGTH)]
            for idx, id_word_representation in enumerate(padded_summary):
                padded_summary_of_one_hots[idx][id_word_representation] = 1

            padded_batch.append((padded_bill_of_one_hots, padded_summary_of_one_hots))
        yield padded_batch
        padded_batch = []

    #convert to integers
    print "finished inputting bills and summaries"
