# array('output')
def ngrams(input, n):
    input = input.split(' ')
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    if n == 1 :
        print (n)
        print (output)
    else:
        print (n)
        print (output)
        return ngrams('abu like to read while he like to sleep', n-1)
    
ngrams('abu like to read while he like to sleep', 3)
