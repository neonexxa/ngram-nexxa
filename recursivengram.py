import numpy as np
import json

# PATH = '../Python/nltk-scikitlearn/Research/rawSAX/rajb.txt'


def nlabel(x):
    return {
        1: 'Unigram',
        2: 'Bigram',
        3: 'Trigram',
        4: '4-gram',
        5: '5-gram',
        6: '6-gram',
        7: '7-gram',
        8: '8-gram',
        9: '9-gram',
        10: '10-gram',
    }[x]

# maininput denote the input text/sentence , while n denote the number of of Grams
def ngrams(maininput, n):
    input = list(maininput)
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    # print (nlabel(n) + str(output))
    # print (output)
    values_json = json.dumps(output, sort_keys=True, separators=(',', ':'))
    # print(nlabel(n) + str(values_json))
    return values_json
    # if want recursive
    # if n!=1 :
    #     return ngrams(maininput, n-1)


def getfile(PATH,graminput,byyear,vline):
    print(byyear)
    if (byyear):
        with open(PATH, 'r') as f:
            line = f.readlines()
            print(vline)
            pretext = line[vline]
            fulltext = ''
            removenewline = pretext.replace('\n', '') #remove newline
            cleantext = removenewline.replace('\r','')  #remove windows indicator
            singletext = cleantext.replace(' ', '')  #remove space
            fulltext+=str(singletext)
            return ngrams(fulltext,graminput)
    else:
        with open(PATH, 'rb') as f:
            # ngrams(f.read(), 10)
            dataset = f.readlines()
            # dataset = f.read()
            # print(''.join(map(bytes.decode, dataset)))
            # strdataset = dataset.replace('\n', '')
            # print (strdataset)
            fulltext = ''
            for line in map(bytes.decode, dataset):
                removenewline = line.replace('\n', '') #remove newline
                cleantext = removenewline.replace('\r','')  #remove windows indicator
                singletext = cleantext.replace(' ', '')  #remove space
                fulltext+=str(singletext)
            # print (fulltext)
            ngrams(fulltext,graminput)
            # print(map(bytes.decode, dataset))
            # strdataset = '\n'.join([line.strip() for line in map(bytes.decode, dataset)])
            # print(ngrams(strdataset, 10))
            # haha = np.concatenate(dataset).astype(None)
            # print(haha)
            # for line in dataset:
            #     year = line.split()
            #     print(year)
            #     print(len(line))