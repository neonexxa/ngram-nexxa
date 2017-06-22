def ngrams(input, n):
    input = input.split(' ')
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    return output
    # if n == 1 :
    #     return output
    # else:
    #     return ngrams('abu like to read while he like to sleep', n-1)
    
print(ngrams('abu like to read while he like to sleep', 2))