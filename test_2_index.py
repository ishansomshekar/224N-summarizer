import numpy as np
import ast
from scipy.signal import argrelextrema
from evaluate_prediction import normalize_answer

gold_summaries_file = "summaries_dev_bills_4_150.txt"
gold_indices_file = "indices_dev_bills_4_150.txt"

pred_file = "preds_adam_2nd.txt"

def evaluate():
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')

    with open(pred_file) as f:
        count = 0
        while True:

            first = f.readline()
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            #np_start_preds = np.asarray(a).astype(np.float)

            b = ast.literal_eval(f.readline())
            #np_end_preds = np.asarray(a).astype(np.float)
            f.readline()

            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            gold = gold_indices.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            gold_end = int(gold[1])

            np_start_preds = np.asarray(a)
            start_maxima = argrelextrema(np_start_preds, np.greater)[0]
            tuples = [(x, np_start_preds[x]) for x in start_maxima]
            # print tuples
            start_maxima = sorted(tuples, key = lambda x: x[1])
            # print maxima
            if len(start_maxima) > 0:
                start_index = start_maxima[-1][0]
            else:
                start_index = start_preds.index(max(start_preds))

            np_end_preds = np.asarray(b)
            end_maxima = argrelextrema(np_end_preds, np.greater)[0]
            # print "###########"
            # print end_maxima
            tuples = [(x, np_end_preds[x]) for x in end_maxima]
            # print tuples
            end_maxima = sorted(tuples, key = lambda x: x[1])
            # print maxima
            if len(end_maxima) > 0:
                end_index = end_maxima[-1][0]
            else:
                end_index = end_preds.index(max(end_preds))

            # print
            # print "gold start ", (gold_start)
            # print "our start " , (start_index)
            # print "gold end ", (gold_end)
            # print "our end ", (end_index)

            if (end_index <= start_index):
                count += 1

            # text = gold_standard_summaries.readline()
            # summary = ' '.join(text.split()[start_index:end_index])
            # gold_summary = ' '.join(text.split()[gold_start:gold_end])
            # summary = normalize_answer(summary)
            # gold_summary = normalize_answer(gold_summary)

            # f.write('\n')
            # f.write(summary + ' \n')
            # f.write(gold_summary + ' \n')

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)

            if start_index == gold_start:
                start_num_exact_correct += 1
            if end_index == gold_end:
                end_num_exact_correct += 1
            
            number_indices += 1
            correct_preds += overlap
            total_preds += len(x)
            total_correct += len(y)

            start_exact_match = start_num_exact_correct/number_indices
            end_exact_match = end_num_exact_correct/number_indices
            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print "evaluate:   "
    print start_exact_match, end_exact_match, p, r, f1
    print "these had invalid end indices: ", count

def localminimaworestriction():
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')

    with open(pred_file) as f:
        count = 0
        while True:
            # sp = f.readline()
            # if not sp:
            #     break
            first = f.readline()
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            start_maxima = argrelextrema(a, np.greater)[0]
            tuples = [(x, a[x]) for x in start_maxima]
            # print tuples
            start_maxima = sorted(tuples, key = lambda x: x[1])
            # print maxima
            if len(start_maxima) > 0:
                a_idx = start_maxima[-1][0]
            else:
                a_idx = np.argmax(a)

            end_maxima = argrelextrema(b, np.greater)[0]
            # print "###########"
            # print end_maxima
            tuples = [(x, b[x]) for x in end_maxima]
            # print tuples
            end_maxima = sorted(tuples, key = lambda x: x[1])
            # print maxima
            if len(end_maxima) > 0:
                b_idx = end_maxima[-1][0]
            else:
                b_idx = np.argmax(b)

            if b_idx <= a_idx:
                count += 1

            # text = gold_standard_summaries.readline()
            # summary = ' '.join(text.split()[start_index:end_index])
            # gold_summary = ' '.join(text.split()[gold_start:gold_end])
            # summary = normalize_answer(summary)
            # gold_summary = normalize_answer(gold_summary)

            # print(f.readline())
            # print(f.readline())
            # print(f.readline())
            # print(f.readline())

            gold = gold_indices.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
            if start_index == gold_start:
                start_num_exact_correct += 1
            if end_index == gold_end:
                end_num_exact_correct += 1
            
            number_indices += 1
            correct_preds += overlap
            total_preds += len(x)
            total_correct += len(y)

    start_exact_match = start_num_exact_correct/number_indices
    end_exact_match = end_num_exact_correct/number_indices
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print "local minima without restriction:   "
    print start_exact_match, end_exact_match, p, r, f1
    print "these had invalid end indices: ", count

def localminima():
    gold_summaries = open(gold_summaries_file, 'r')
    #bills = open('bills_dev_bills_5_250.txt', 'r')
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')

    #summaries_consolidated = open("summaries_from_preds", "w")
    lengths = []
    with open(pred_file) as f:
        while True:
            # sp = f.readline()
            # if not sp:
            #     break
            first = f.readline()
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            start_maxima = argrelextrema(a, np.greater)[0]
            tuples = [(x, a[x]) for x in start_maxima]
            # print tuples
            start_maxima = sorted(tuples, key = lambda x: x[1])
            # print maxima
            if len(start_maxima) > 0:
                a_idx = start_maxima[-1][0]
            else:
                a_idx = np.argmax(a)

            end_maxima = argrelextrema(b, np.greater)[0]
            # print "###########"
            # print end_maxima
            tuples = [(x, b[x]) for x in end_maxima if x > a_idx]
            # print tuples
            end_maxima = sorted(tuples, key = lambda x: x[1])
            # print maxima
            if len(end_maxima) > 0:
                b_idx = end_maxima[-1][0]
            else:
                b_idx = np.argmax(b)

            # text = bills.readline()
            # summary = ' '.join(text.split()[a_idx:b_idx])
            # gold_summary = gold_summaries.readline()
            # summary = normalize_answer(summary)
            # gold_summary = normalize_answer(gold_summary)
            # summaries_consolidated.write(summary + "\n")
            # summaries_consolidated.write(gold_summary + "\n")

            # print(f.readline())
            # print(f.readline())
            # print(f.readline())
            # print(f.readline())

            gold = gold_indices.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)

            lengths.append(end_index - start_index)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
            if start_index == gold_start:
                start_num_exact_correct += 1
            if end_index == gold_end:
                end_num_exact_correct += 1
            
            number_indices += 1
            correct_preds += overlap
            total_preds += len(x)
            total_correct += len(y)

    start_exact_match = start_num_exact_correct/number_indices
    end_exact_match = end_num_exact_correct/number_indices
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print "local minima:   "
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)



def neither_fixed():
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(pred_file) as f:
        while True:
            # sp = f.readline()
            # if not sp:
            #     break
            first = f.readline()
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            a_idx = len(a) - 2
            b_idx = len(b) - 1

            b_max = b_idx
            total_max = a[a_idx] * b[b_max]

            for i in xrange(len(a)-3, -1, -1):
                if b[i + 1] > b[b_max]:
                    b_max = i + 1
                if a[i] * b[b_max] > total_max:
                    a_idx = i
                    b_idx = b_max

            gold = gold_indices.readline()
            gold = gold.split()
            # print gold
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)
            lengths.append(end_index - start_index)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
            if start_index == gold_start:
                start_num_exact_correct += 1
            if end_index == gold_end:
                end_num_exact_correct += 1
            
            number_indices += 1
            correct_preds += overlap
            total_preds += len(x)
            total_correct += len(y)

    start_exact_match = start_num_exact_correct/number_indices
    end_exact_match = end_num_exact_correct/number_indices
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print "neither_fixed:"
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)



    print "########################################################"    

def end_max_fixed():
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(pred_file) as f:
        while True:
            # sp = f.readline()
            # if not sp:
            #     break
            first = f.readline()
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            b_idx = np.argmax(b)
            #if out of bounds, fix it
            a_idx = b_idx
            while a_idx >= b_idx:
                a_idx = np.argmax(a)
                a[a_idx] = 0

            lengths.append(b_idx - a_idx)

            gold = gold_indices.readline()
            gold = gold.split()
            # print gold
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
            if start_index == gold_start:
                start_num_exact_correct += 1
            if end_index == gold_end:
                end_num_exact_correct += 1
            
            number_indices += 1
            correct_preds += overlap
            total_preds += len(x)
            total_correct += len(y)

    start_exact_match = start_num_exact_correct/number_indices
    end_exact_match = end_num_exact_correct/number_indices
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print "end_max_fixed:"
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)

    print "########################################################" 

def max_fixed():
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(pred_file) as f:
        while True:
            # sp = f.readline()
            # if not sp:
            #     break
            first = f.readline()
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            a_idx = np.argmax(a)
            #if out of bounds, fix it
            b_idx = a_idx
            while b_idx <= a_idx:
                b_idx = np.argmax(b)
                b[b_idx] = 0

            lengths.append(b_idx - a_idx)

            gold = gold_indices.readline()
            gold = gold.split()
            # print gold
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
            if start_index == gold_start:
                start_num_exact_correct += 1
            if end_index == gold_end:
                end_num_exact_correct += 1
            
            number_indices += 1
            correct_preds += overlap
            total_preds += len(x)
            total_correct += len(y)

    start_exact_match = start_num_exact_correct/number_indices
    end_exact_match = end_num_exact_correct/number_indices
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print "start_max_fixed:"
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)

    print "########################################################" 

evaluate()
localminima()
neither_fixed()
max_fixed()
end_max_fixed()



# a = [2.848939554134831, -4.428165515496098, -4.5981836049053, -4.951680759743077, -4.797728786194734, -6.075592494767344, -4.455245031209476, -4.210303988869629, -0.18460147995744236, -2.9590912155454343, -4.880254828974203, -5.003931779212303, -4.6942268998234224, -4.115971683046096, -4.532994396351281, -4.006382777881627, -3.344002807574128, -4.93508223278937, -4.675240352065947, -1.382916938447767, -5.346555131698352, -5.024131089467739, -5.219344967376719, 1.1469164278983208, -5.144172853173249, -3.4756884558719348, -4.755150874009418, -4.913049795690335, -4.796680243455004, -5.193038075979015, -4.482296271700615, -3.651116135275309, -4.463303124531855, -6.192565777046916, -4.822039751105489, -4.393995715521606, -5.234483397433182, -4.334939516290648, -4.375977468623231, -3.844782357734566, -3.84346434447539, -4.109691932941312, -6.881172424156361, -5.545001883470713, -5.242214093794978, -1.0962528216310095, -6.226497157205387, -5.458925198204767, -6.828011103019948, -1.3435619049540974, -5.211403085907184, -3.8987871218101535, -5.415991413080206, -3.9246735642346486, -3.508676292247281, -4.389991630051236, -5.216577113744826, -6.353926088930323, -3.524118300227589, -5.783332601500335, -2.7873707748285965, -2.530097707710479, -3.7086514052012642, -4.961250671059997, -3.814656084627327, -5.554954019740926, -6.745244914947717, -4.965128675114845, -6.120808041308469, -4.632403831879248, -5.59416475888502, -4.943096273320714, -4.238337398749395, -4.42887070728017, -5.019214449429757, -4.962612830529366, -5.320331548777242, -5.4113560353711945, -6.360683735779013, -4.97880352418067, -1.378810896063275, -4.699075507441333, -6.736814426920038, -4.032126482571054, -6.474212512821866, -5.114946928689419, -5.887872455688111, -3.593653991946664, -2.2800621734447124, -2.6046272043167935, -2.3176720320335784, -5.198464603204213, -3.139167060641748, -5.74559194897083, -4.517619576791511, -4.0623423356733, -5.830342411045281, -5.5758497504934, -6.037139546145668, -4.759462469850638, -4.328245699490613, -5.410474771405902, -3.3582450206233494, -2.587299565273053, -5.185049820384556, -6.252506912026105, -5.812972045091255, -6.373430079820304, -6.040991291479149, -5.481534353859259, -8.617922144638626, -8.350890186728074, -9.166215696689388, -8.219850338404084, -8.396191002264603, -9.539282130008527, -10.438045937034127, -9.874564372497304, -10.352496368183248, -9.8275225892924, -9.318483335948565, -9.758605591643637, -9.564731678732485, -9.630198164854516, -9.407484495986967, -10.055304657784685, -10.147616944930395, -10.113436497049861, -9.783212451345657, -10.048293400730037, -10.284363603937642, -10.517210922864052, -11.128413597681652, -10.87704776805857, -10.191677634335306, -10.374711528241962, -10.74068131434887, -10.434043799899303, -10.6766616453456, -11.011152404547143, -10.873348709071875, -10.6766616453456, -10.840336893896247, -10.570270249313412, -10.628859182003325, -10.628859182003325, -10.628859182003325, -10.628859182003325, -10.628859182003325, -10.628859182003325, -10.628859182003325]
# b = [-5.299629441058533, -5.4754403658197734, -5.4241176128890904, -10.330974571128724, -5.576172480784933, -6.23288014024061, -10.025506370981404, -6.330684455279755, -6.720862221183987, -6.138776827648013, -5.5894716279483445, -4.985269134652707, -5.331427428463419, -5.222817934900142, -10.285069705311955, -4.9916229828080265, -5.219581922975013, -5.415930038859449, -9.388340881766918, -3.326851870427224, -9.572303067056339, -4.135393670002575, -9.777927706320359, -2.5545099724495373, -5.981048163287067, -5.309138989263497, -6.244650209136491, -10.332917617578433, -6.431427441453655, -8.024641726222853, -5.368375055183365, -7.15094762125668, -5.533658610846853, -6.858219833519078, -7.009721060225951, -5.278032867857824, -4.787479867705719, -5.190506181854476, -5.420129935909232, -10.378926365427635, -5.227947574376284, -4.917682310662953, -7.953643442058221, -5.237550341053588, -9.480834338570977, -3.333647593468236, -9.266207012123544, -4.0952612503895836, -6.427637687447223, -6.63465954627649, -9.126927672108652, -5.125419075911229, -6.390284306201075, -4.85061670311981, -5.4778537173895625, -8.136872377509471, -10.189118933854854, -4.030929516587945, -6.531138282892977, -6.911851659182757, -5.5526050091527175, -5.48973697068074, -6.8566444232570944, -6.827501017293207, -5.193761468209518, -5.177288906348595, -10.19241407441686, -4.939182571940905, -5.938537204333617, -10.34381198885419, -6.412865049023413, -7.141562452637356, -6.266035239584228, -5.059043585730137, -6.486002933120694, -5.777313772271346, -8.438275351106292, -9.140527838014375, -4.067651082926067, -4.071209215131411, -6.7037021692297785, -5.937196783661223, -7.008826404436265, -5.751741243316383, -8.65587811476427, -5.289969589838866, -6.519662854682675, -4.144826054755839, -6.623224226660382, -5.339766706224185, -3.0195607595194653, -10.08957609052414, -6.191775599196484, -9.800213737158455, -4.50406586667506, -6.379300180310715, -9.443538838819652, -4.37221652695235, -3.963844436813779, -9.968876540732868, -5.143908642356972, -6.472385119170353, -4.693761531169031, -5.366030832218128, -8.475914219290509, -10.163287083494124, -4.18864626615665, -7.72940459613026, -3.9148604869657087, -9.489249459044554, -5.374693728564534, -5.586320993279604, -5.971550361676288, -5.392494355041074, -5.7913760944286015, -5.66972395353765, -5.9843353128751, -5.905922385903176, -6.207686122973191, -5.963680273644117, -6.083869049965108, -6.102539082427192, -6.169027253324088, -5.915712189960399, -5.948589560897533, -6.1100035233515415, -5.852169550285114, -5.933143556655014, -5.956222360872367, -6.051481486593734, -6.4718391138551565, -6.3216249963442195, -6.251980191125509, -6.5693858422281854, -6.395586999646527, -6.3964437698798395, -6.498053342572212, -6.479012595445721, -6.731194090725552, -6.8418529669007055, -6.830941782035402, -6.731194090725552, -6.886801661321546, -6.785388632259206, -7.13791439800929, -7.13791439800929, -7.13791439800929, -7.13791439800929, -7.13791439800929, -7.13791439800929, -7.13791439800929]
# a = np.exp(a - np.amax(a))
# a = a / np.sum(a)
# b = np.exp(b - np.amax(b))
# b = b / np.sum(b)

# a_idx = len(a) - 2
# b_idx = len(b) -1

# b_max = b_idx
# total_max = a[a_idx] * b[b_max]

# for i in xrange(len(a)-3, -1, -1):
#     if b[i + 1] > b[b_max]:
#         b_max = i + 1
#     if a[i] * b[b_max] > total_max:
#         a_idx = i
#         b_idx = b_max

       
# print (a_idx, b_idx)
# print a[a_idx], b[b_idx]
# print a[0], b[94]