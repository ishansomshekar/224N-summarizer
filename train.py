from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os

all_bill_directory = '/CLEANINGTEST/'
all_summary_directory = '/ALL_GOLD_SUMMARIES/'
BATCH_SIZE = 3
MAX_SUMMARY_LENGTH = 90
MAX_BILL_LENGTH = 500

def main():

    bills_datapath = os.getcwd() + all_bill_directory
    gold_summaries_datapath = os.getcwd() + all_summary_directory

    embedding_wrapper = EmbeddingWrapper(bills_datapath)
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()

    f_generator = file_generator(BATCH_SIZE)

    #pad the bills and summaries
    padded_batch = []
    for bill_batch, summary_batch in f_generator:
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


    #convert to integers


if __name__ == "__main__":
    main()

