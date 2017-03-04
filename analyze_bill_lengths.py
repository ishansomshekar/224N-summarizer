import csv
import statistics
import numpy as np

def main():

    summary_lengths = []
    bill_lengths = []

    with open("data_2x_greater_than_summaries.csv", 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            bill_lengths.append(int(row[2]))
            summary_lengths.append(int(row[4]))

    bill_lengths = np.array(bill_lengths)
    bill_median = statistics.median(bill_lengths)
    bill_variance = statistics.variance(bill_lengths)
    bill_std = statistics.stdev(bill_lengths)
    bill_max = max(bill_lengths)
    print "bill median ", bill_median
    print "bill standard deviation ", bill_std
    print "bill max", bill_max

    percentile_90 = np.percentile(bill_lengths, 90)
    print "90th ", percentile_90
    percentile_85 = np.percentile(bill_lengths, 85)
    print "85th ", percentile_85
    percentile_75 = np.percentile(bill_lengths, 75)
    print "75th ", percentile_75
    percentile_65 = np.percentile(bill_lengths, 65)
    print "65th ", percentile_65


    summary_lengths = np.array(summary_lengths)
    summary_median = statistics.median(summary_lengths)
    summary_variance = statistics.variance(summary_lengths)
    summ_std = statistics.stdev(summary_lengths)
    summ_max = max(summary_lengths)
    print "summ median ", summary_median
    print "summ variance ", summary_variance
    print "summ standard deviation ", summ_std
    print "summ max ", summ_max 

    percentile_90 = np.percentile(summary_lengths, 90)
    print "90th ", percentile_90
    percentile_85 = np.percentile(summary_lengths, 85)
    print "85th ", percentile_85
    percentile_75 = np.percentile(summary_lengths, 75)
    print "75th ", percentile_75
    percentile_65 = np.percentile(summary_lengths, 65)
    print "65th ", percentile_65


if __name__ == "__main__":
    main()
