from read_in_datafile import file_generator
from embedding_wrapper import EmbeddingWrapper

def batch_generator(embedding_wrapper, bill_data_path, indices_data_path, sequences_data_path, batch_size, MAX_BILL_LENGTH):

    f_generator = file_generator(batch_size, bill_data_path, indices_data_path, sequences_data_path)

    #pad the bills and summaries
    print "now padding and encoding batches"
    padded_bills = []
    padded_start_indices = []
    padded_end_indices = []
    padded_masks = []
    for bill_batch, indices_batch, sequences in f_generator:
        # print "batch"
        # print bill_batch
        #print indices_batch
        for idx, bill in enumerate(bill_batch):
            start_index, end_index = indices_batch[idx]
            padded_bill = [embedding_wrapper.get_value(word) for word in bill]
            # padded_summary = [embedding_wrapper.get_value(word) for word in summary] d g
            mask = [True] * min(len(bill), MAX_BILL_LENGTH)
            padded_bill = padded_bill[:MAX_BILL_LENGTH]
            # padded_summary = padded_summary[:MAX_SUMMARY_LENGTH]
            mask = mask[:MAX_BILL_LENGTH]

            for i in xrange(0, MAX_BILL_LENGTH - len(padded_bill)):
                padded_bill.append(embedding_wrapper.get_value(embedding_wrapper.pad))
                mask.append(False)

            padded_masks.append(mask)
            padded_bills.append(padded_bill)

            start_index_one_hot = [0] * MAX_BILL_LENGTH
            if start_index >= MAX_BILL_LENGTH:
                start_index_one_hot[0] = 1
            else:
                start_index_one_hot[start_index] = 1
            
            end_index_one_hot = [0] * MAX_BILL_LENGTH
            if end_index >= MAX_BILL_LENGTH:
                end_index_one_hot[MAX_BILL_LENGTH - 1] = 1
            else:
                end_index_one_hot[end_index] = 1

            padded_start_indices.append(start_index_one_hot)
            padded_end_indices.append(end_index_one_hot)

        yield padded_bills, padded_start_indices, padded_end_indices, padded_masks, sequences
        padded_bills = []
        padded_start_indices = []
        padded_end_indices = []
        padded_masks = []

    #convert to integers
    print "finished inputting bills and summaries"
