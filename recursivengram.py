def nlabel(x):
    return {
        1: 'Unigram',
        2: 'Bigram',
        3: 'Trigram',
    }[x]

# maininput denote the input text/sentence , while n denote the number of of Grams
def ngrams(maininput, n):
    input = maininput.split(' ')
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    print (nlabel(n) + str(output))
    if n!=1 :
        return ngrams(maininput, n-1)
    
ngrams('abu like to read while he like to sleep', 3)
