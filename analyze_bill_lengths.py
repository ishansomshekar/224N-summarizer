import csv
import statistics
import numpy as np
from analyze_index_data import make_hist
import matplotlib.pyplot as plt

def main():

    summary_lengths = []
    bill_lengths = []

    with open("bills_modified_only_below_400.csv", 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            bill_lengths.append(int(row[2]))
            summary_lengths.append(int(row[6]))

    bill_lengths = np.array(bill_lengths)
    bill_median = statistics.median(bill_lengths)
    bill_variance = statistics.variance(bill_lengths)
    bill_std = statistics.stdev(bill_lengths)
    bill_max = max(bill_lengths)
    bill_min = min(bill_lengths)
    print "bill median ", bill_median
    print "bill standard deviation ", bill_std
    print "bill max", bill_max
    print "bill min", bill_min


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
    summ_min = min(summary_lengths)
    print "summ median ", summary_median
    print "summ variance ", summary_variance
    print "summ standard deviation ", summ_std
    print "summ max ", summ_max 
    print "summ min ", summ_min

    percentile_90 = np.percentile(summary_lengths, 90)
    print "90th ", percentile_90
    percentile_85 = np.percentile(summary_lengths, 85)
    print "85th ", percentile_85
    percentile_75 = np.percentile(summary_lengths, 75)
    print "75th ", percentile_75
    percentile_65 = np.percentile(summary_lengths, 65)
    print "65th ", percentile_65

    fig, ax = plt.subplots(figsize=(14,5))
    make_hist(ax, [1,1,1,0,0,0], extra_y=1, text_offset=0.1)
    make_hist(ax, bill_lengths, bins=list(range(0,400,50))+ [np.inf], extra_y=6, title = "Bill Length Frequency Diagram", xlabel = "Length", yoffset = 1500)
    plt.show()

    fig, ax = plt.subplots(figsize=(14,5))
    make_hist(ax, [1,1,1,0,0,0], extra_y=1, text_offset=0.1)
    make_hist(ax, summary_lengths, bins=list(range(0,300,50))+ [np.inf], extra_y=6, title = "Bill Length Frequency Diagram", xlabel = "Length", yoffset = 1500)
    plt.show()


if __name__ == "__main__":
    main()
