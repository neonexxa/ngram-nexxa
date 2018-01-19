
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--ln", required = True, help = "location name")
ap.add_argument("-d", "--dn", required = True, help = "data name")
ap.add_argument("-g", "--gn", required = True, help = "ngram name")
args = vars(ap.parse_args())


# In[ ]:


locationame = args["ln"]
dataname = args["dn"]
n = int(args["gn"])
data = {}
data[dataname] = pd.read_csv('rawSAX/'+locationame+'/'+dataname.lower()+locationame.lower()+'.txt', sep=" ", header=None)


# In[ ]:


def sequecingthevalue(thedata):
    output = {}
    for i in range(len(thedata)-n+1):
        g = ' '.join(thedata[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    return output


# In[ ]:


# for printing ngram values
# text_file = open('rawSAX/'+locationame+'/CH.text", "a")
for line in range(len(data[dataname])):
    thedata = data[dataname].loc[line].values[:12]
    output = {}
    for i in range(len(thedata)-n+1):
        g = ' '.join(thedata[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    values_json = json.dumps(output, sort_keys=True, separators=(',', ':'))
    file_path = 'rawSAX/'+locationame+'/'+locationame+dataname+'_'+str(n)+'_gram.json'
    import os.path
    if(os.path.exists(file_path)):
        textfile = open(file_path, 'a')
        textfile.write(values_json+"\n")
    else:
        textfile = open(file_path, 'w')
        textfile.write(values_json+"\n")
#     print(type(values_json))
#     # print(nlabel(n) + str(values_json))
    textfile.close()
    print(values_json)
# text_file.close()
# pd.read_json(values_json, typ='series')


