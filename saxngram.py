import recursivengram as rngm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required = True, help = "collection/location/specific")
ap.add_argument("-s", "--specific", required = False, help = "by specific. example: rach")
args = vars(ap.parse_args())

loc = ['ch','jb','kch','lgk','pj','sbg']
locname = ['cheras','johor baru','kucing','lgk','pj','sbg']
datasettype = ['ra','rf','tem']
datasetname = ['Rain amount','Rain Frequency','Temperature']

graminput = 2

if (args["type"] == "collection"):
	print ("============================================================")
	print ("Ngram by collection")
	for y in range(len(datasettype)):
		print ("============================================================")
		print (datasetname[y])
		for x in range(len(loc)):
			print (loc[x].upper())
			PATH = 'rawSAX/'+loc[x].upper()+'/'+datasettype[y]+loc[x]+'.txt'
			rngm.getfile(PATH,graminput)
elif (args["type"] == "location"):
	print ("============================================================")
	print ("Ngram by Location")
	for y in range(len(loc)):
		print ("============================================================")
		print (locname[y])
		for x in range(len(datasettype)):
			print (datasetname[x])
			PATH = 'rawSAX/'+loc[y].upper()+'/'+datasettype[x]+loc[y]+'.txt'
			rngm.getfile(PATH,graminput)
elif (args["type"] == "specific"):
	col = args["specific"][:2]
	loct = args["specific"][2:]
	colpos = datasettype.index(col)
	locpos = loc.index(loct)
	print ("============================================================")
	print ("Ngram for "+datasetname[colpos]+" at "+locname[locpos])
	print ("============================================================")
	PATH = 'rawSAX/'+loct.upper()+'/'+col+loct+'.txt'
	rngm.getfile(PATH,graminput)


