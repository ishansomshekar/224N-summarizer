# generator

def batch_generator(embedding_wrapper, bill_name_file, batch_size, MAX_BILL_LENGTH, MAX_SUMMARY_LENGTH):

    f_generator = file_generator(batch_size, bill_name_file)

    #pad the bills and summaries
    print "now padding and encoding batches"
    padded_batch = []
    for bill_batch, summary_batch in f_generator:
        print "batch"
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

            padded_batch.append((padded_bill, padded_summary))
        yield padded_batch
        padded_batch = []

    #convert to integers
    print "finished inputting bills and summaries"
