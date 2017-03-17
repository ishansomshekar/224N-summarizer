import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import os
import sys
import csv
from random import randint
import statistics

def make_hist(ax, x, bins=None, binlabels=None, width=0.85, extra_x=1, extra_y=4, 
              text_offset=0.3, title=r"Frequency diagram", 
              xlabel="Values", ylabel="Frequency"):
    if bins is None:
        xmax = max(x)+extra_x
        bins = range(xmax+1)
    if binlabels is None:
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(bins[i], bins[i+1]-1)
                         for i in range(len(bins)-1)]
        else:
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(*bins[i:i+2])
                         for i in range(len(bins)-1)]
        if bins[-1] == np.inf:
            binlabels[-1] = '{}+'.format(bins[-2])
    n, bins = np.histogram(x, bins=bins)
    patches = ax.bar(range(len(n)), n, align='center', width=width)
    ymax = max(n)+extra_y

    ax.set_xticks(range(len(binlabels)))
    ax.set_xticklabels(binlabels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis='y')
    # http://stackoverflow.com/a/28720127/190597 (peeol)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # http://stackoverflow.com/a/11417222/190597 (gcalmettes)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    autolabel(patches, text_offset)

def autolabel(rects, shift=0.3):
    """
    http://matplotlib.org/1.2.1/examples/pylab_examples/barchart_demo.html
    """
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            plt.text(rect.get_x()+rect.get_width()/2., height+shift, '%d'%int(height),
                     ha='center', va='bottom')

def plot_start_index_histogram(start_indexes):
    fig, ax = plt.subplots(figsize=(14,5))
    # num_zeros = end_indexes.count(0)
    # print float(num_zeros)/len(end_indexes)
    make_hist(ax, [1,1,1,0,0,0], extra_y=1, text_offset=0.1)
    make_hist(ax, start_indexes, bins=list(range(0,140,10))+ [np.inf], extra_y=6, title = "Start Index Frequency Diagram", xlabel = "Index")
    plt.show()

def plot_end_index_histogram(end_indexes):
    fig, ax = plt.subplots(figsize=(14,5))
    # num_zeros = end_indexes.count(0)
    # print float(num_zeros)/len(end_indexes)
    make_hist(ax, [1,1,1,0,0,0], extra_y=1, text_offset=0.1)
    make_hist(ax, end_indexes, bins=list(range(0,340,30))+ [np.inf], extra_y=6, title = "End Index Frequency Diagram", xlabel = "Index")
    plt.show()

def plot_random_sample_start_and_end(start_indexes, end_indexes):
    num_of_lines = 3

    # Defines a coluor for each line
    colours = ['c', 'crimson', 'chartreuse', 'blue'] 

    # Defines a marker for each line
    markers = ['o', 'v', '*']

    indices = set()
    while len(indices) != 100:
        indices.add(randint(0,88288))


    xes = xrange(1, len(indices) + 1)
    indices_list = list(indices)
    starts = [start_indexes[i] for i in indices_list]
    ends = [end_indexes[i] for i in indices_list]

    plt.scatter(starts, xes, marker=markers[0])
    plt.scatter(ends, xes, marker = markers[1])

    start_x_lines = zip(starts,xes)
    end_x_lines = zip(ends,xes)
    for i in xrange(0, len(starts)):
        plt.plot([starts[i], ends[i]], [i + 1,i + 1], c=colours[3])

    # Show grid in the plot
    plt.grid()
    # Finally, display the plot
    plt.ylim([-10,len(starts) + 10])
    plt.xlim([0, max(ends) + 10])
    plt.title("Start and End Indices for Random Sample of 100 Bills")
    plt.xlabel('Indices')
    plt.ylabel('Bills')
    plt.xticks(np.arange(-10, max(ends)+10, 10.0))
    plt.yticks(np.arange(0, len(starts)+10, 50.0))
    plt.show()

def main():
    csv_file = 'bills_modified_only_below_400.csv'
    start_indexes = []
    end_indexes = []

    with open(csv_file, 'rb') as csvfile:
            rows = [row for row in csv.reader(csvfile.read().splitlines())]
            for row in rows:    
                file_name = row[0]
                start_indexes.append(int(row[4]))
                end_indexes.append(int(row[5]))
    
    start_indices = np.array(start_indexes)
    start_median = statistics.median(start_indices)
    start_variance = statistics.variance(start_indices)
    start_std = statistics.stdev(start_indices)
    start_max = max(start_indices)
    start_min = min(start_indices)
    print "START INDICES"
    print "start index median ", start_median
    print "start index standard deviation ", start_std
    print "start index max", start_max
    print "start index min", start_min

    percentile_90 = np.percentile(start_indices, 90)
    print "90th ", percentile_90
    percentile_85 = np.percentile(start_indices, 85)
    print "85th ", percentile_85
    percentile_75 = np.percentile(start_indices, 75)
    print "75th ", percentile_75
    percentile_65 = np.percentile(start_indices, 65)
    print "65th ", percentile_65

    print "END INDICES"
    end_indices = np.array(end_indexes)
    end_median = statistics.median(end_indices)
    end_variance = statistics.variance(end_indices)
    end_std = statistics.stdev(end_indices)
    end_max = max(end_indices)
    end_min = min(end_indices)
    
    print "end index median ", end_median
    print "end index standard deviation ", end_std
    print "end index max", end_max
    print "end index min", end_min

    percentile_90 = np.percentile(end_indices, 90)
    print "90th ", percentile_90
    percentile_85 = np.percentile(end_indices, 85)
    print "85th ", percentile_85
    percentile_75 = np.percentile(end_indices, 75)
    print "75th ", percentile_75
    percentile_65 = np.percentile(end_indices, 65)
    print "65th ", percentile_65

    plot_random_sample_start_and_end(start_indexes, end_indexes)
    plot_end_index_histogram(end_indexes)
    plot_start_index_histogram(start_indexes)


if __name__ == "__main__":
    main()
