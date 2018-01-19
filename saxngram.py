import recursivengram as rngm
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required = True, help = "collection/location/specific/corrosion")
ap.add_argument("-y", "--year", required = False, help = "byyear")
ap.add_argument("-s", "--specific", required = False, help = "by specific. example: rach")
ap.add_argument("-cr", "--corrosion", required = False, help = "by corrosion. example: cl/iron/pH/salt/temp")
ap.add_argument("-tr", "--train", required = False, help = "yes/no")
args = vars(ap.parse_args())

loc = ['ch','jb','kch','lgk','pj','sbg']
locname = ['cam','johor baru','kucing','lgk','pj','sbg']
datasettype = ['ra','rf','tem']
datasetname = ['Rain amount','Rain Frequency','Temperature']

graminput = 2
byyear = 0

if(args["year"] == "byyear"):
	byyear = 1
	if (args["type"] == "collection"):
		print ("============================================================")
		print ("Ngram by collection")
		for y in range(len(datasettype)):
			print ("============================================================")
			print (datasetname[y])
			for x in range(len(loc)):
				print (loc[x].upper())
				PATH = 'rawSAX/'+loc[x].upper()+'/'+datasettype[y]+loc[x]+'.txt'
                print(PATH)
				rngm.getfile(PATH,graminput,byyear)
	elif (args["type"] == "location"):
		print ("============================================================")
		print ("Ngram by Location")
		for y in range(len(loc)):
			print ("============================================================")
			print (locname[y])
			if(args["train"] == "yes"):
				tot = '/test/ts_'
				if(y==0) or (y==2):
					ran = 7
				elif(y==1) or (y==4) or (y==5):
					ran = 8
				elif(y==3):
					ran = 5
			else:
				tot = '/'
				if(y==0) or (y==2) or (y==4):
					ran = 17
				elif(y==1) or (y==5):
					ran = 18
				elif(y==3):
					ran = 14
			for v in range(ran):
				storagekey = []
				for x in range(len(datasettype)):
					print (datasetname[x])
					PATH = 'rawSAX/'+loc[y].upper()+tot+datasettype[x]+loc[y]+'.txt'
					print(str(y)+str(v)+str(x))
					storagekey.append(rngm.getfile(PATH,graminput,byyear,v))
				(a_b,a_c,b_a,b_b,b_c,b_d,b_e,c_a,c_b,c_c,c_d,c_e,d_b,d_c,d_d,d_e,e_b,e_c,e_d,e_e) = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
				for x in range(3):
					data  = json.loads(storagekey[x])
					if 'a b' in data:
						a_b += data['a b']
					if 'a c' in data:
						a_c += data['a c']
					if 'b a' in data:
						b_a += data['b a']
					if 'b b' in data:
						b_b += data['b b']
					if 'b c' in data:
						b_c += data['b c']
					if 'b d' in data:
						b_d += data['b d']
					if 'b e' in data:
						b_e += data['b e']
					if 'c a' in data:
						c_a += data['c a']
					if 'c b' in data:
						c_b += data['c b']
					if 'c c' in data:
						c_c += data['c c']
					if 'c d' in data:
						c_d += data['c d']
					if 'c e' in data:
						c_e += data['c e']
					if 'd b' in data:
						d_b += data['d b']
					if 'd c' in data:
						d_c += data['d c']
					if 'd d' in data:
						d_d += data['d d']
					if 'd e' in data:
						d_e += data['d e']
					if 'e b' in data:
						e_b += data['e b']
					if 'e c' in data:
						e_c += data['e c']
					if 'e d' in data:
						e_d += data['e d']
					if 'e e' in data:
						e_e += data['e e']
					print(data)
					# newObj[item.key[0]] += item.value;
				# print("new value")
				# print("a_b : "+str(a_b))
				# print("a_c : "+str(a_c))
				# print("b_a : "+str(b_a))
				# print("b_b : "+str(b_b))
				# print("b_c : "+str(b_c))
				# print("b_d : "+str(b_d))
				# print("b_e : "+str(b_e))
				# print("c_a : "+str(c_a))
				# print("c_b : "+str(c_b))
				# print("c_c : "+str(c_c))
				# print("c_d : "+str(c_d))
				# print("c_e : "+str(c_e))
				# print("d_b : "+str(d_b))
				# print("d_c : "+str(d_c))
				# print("d_d : "+str(d_d))
				# print("d_e : "+str(d_e))
				# print("e_b : "+str(e_b))
				# print("e_c : "+str(e_c))
				# print("e_d : "+str(e_d))
				# print("e_e : "+str(e_e))
# 				text_file = open('rawSAX/'+loc[y].upper()+tot+loc[y].upper()+".txt", "a")
# 				text_file.write(str(a_b)+" "+str(a_c)+" "+str(b_a)+" "+str(b_b)+" "+str(b_c)+" "+str(b_d)+" "+str(b_e)+" "+str(c_a)+" "+str(c_b)+" "+str(c_c)+" "+str(c_d)+" "+str(c_e)+" "+str(d_b)+" "+str(d_c)+" "+str(d_d)+" "+str(d_e)+" "+str(e_b)+" "+str(e_c)+" "+str(e_d)+" "+str(e_e)+"\n")
# 				text_file.close()
	elif (args["type"] == "specific"):
		col = args["specific"][:2]
		loct = args["specific"][2:]
		colpos = datasettype.index(col)
		locpos = loc.index(loct)
		print ("============================================================")
		print ("Ngram for "+datasetname[colpos]+" at "+locname[locpos])
		print ("============================================================")
		PATH = 'rawSAX/'+loct.upper()+'/'+col+loct+'.txt'
		rngm.getfile(PATH,graminput,byyear)
	elif (args["type"] == "corrosion"):
		PATH = 'sss/ef'+args["corrosion"]+'.txt'
		filename = open(PATH)
#    		filesize=sum(1 for _ in filename)
		print(PATH,filesize)
		storagekey = []
		for v in range(filesize):
			# print(rngm.getfile(PATH,graminput,byyear,v))
			storagekey.append(rngm.getfile(PATH,graminput,byyear,v))
			# print (storagekey)
		# data = json.loads(storagekey)
		# print(storagekey)
		# print label
# 		text_file = open('sss/ngram_'+args["corrosion"]+".txt", "a")
# 		text_file.write("a_b a_c b_a b_b b_c b_d c_a c_b c_c c_d c_e d_b d_c d_d d_e e_b e_c e_d e_e\n")
# 		text_file.close()
		# end print label

		for x in range(filesize):
			(a_b,a_c,b_a,b_b,b_c,b_d,b_e,c_a,c_b,c_c,c_d,c_e,d_b,d_c,d_d,d_e,e_b,e_c,e_d,e_e) = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
			data  = json.loads(storagekey[x])
			if 'a b' in data:
				a_b += data['a b']
			if 'a c' in data:
				a_c += data['a c']
			if 'b a' in data:
				b_a += data['b a']
			if 'b b' in data:
				b_b += data['b b']
			if 'b c' in data:
				b_c += data['b c']
			if 'b d' in data:
				b_d += data['b d']
			if 'b e' in data:
				b_e += data['b e']
			if 'c a' in data:
				c_a += data['c a']
			if 'c b' in data:
				c_b += data['c b']
			if 'c c' in data:
				c_c += data['c c']
			if 'c d' in data:
				c_d += data['c d']
			if 'c e' in data:
				c_e += data['c e']
			if 'd b' in data:
				d_b += data['d b']
			if 'd c' in data:
				d_c += data['d c']
			if 'd d' in data:
				d_d += data['d d']
			if 'd e' in data:
				d_e += data['d e']
			if 'e b' in data:
				e_b += data['e b']
			if 'e c' in data:
				e_c += data['e c']
			if 'e d' in data:
				e_d += data['e d']
			if 'e e' in data:
				e_e += data['e e']
			print(data)
# 			text_file = open('sss/ngram_'+args["corrosion"]+".txt", "a")
# 			text_file.write(str(a_b)+" "+str(a_c)+" "+str(b_a)+" "+str(b_b)+" "+str(b_c)+" "+str(b_d)+" "+str(b_e)+" "+str(c_a)+" "+str(c_b)+" "+str(c_c)+" "+str(c_d)+" "+str(c_e)+" "+str(d_b)+" "+str(d_c)+" "+str(d_d)+" "+str(d_e)+" "+str(e_b)+" "+str(e_c)+" "+str(e_d)+" "+str(e_e)+"\n")
# 			text_file.close()
else:
	if (args["type"] == "collection"):
		print ("============================================================")
		print ("Ngram by collection")
		for y in range(len(datasettype)):
			print ("============================================================")
			print (datasetname[y])
			for x in range(len(loc)):
				print (loc[x].upper())
				PATH = 'rawSAX/'+loc[x].upper()+'/'+datasettype[y]+loc[x]+'.txt'
				rngm.getfile(PATH,graminput,byyear)
	elif (args["type"] == "location"):
		print ("============================================================")
		print ("Ngram by Location")
		for y in range(len(loc)):
			print ("============================================================")
			print (locname[y])
			for x in range(len(datasettype)):
				print (datasetname[x])
				PATH = 'rawSAX/'+loc[y].upper()+'/'+datasettype[x]+loc[y]+'.txt'
				rngm.getfile(PATH,graminput,byyear)
	elif (args["type"] == "specific"):
		col = args["specific"][:2]
		loct = args["specific"][2:]
		colpos = datasettype.index(col)
		locpos = loc.index(loct)
		print ("============================================================")
		print ("Ngram for "+datasetname[colpos]+" at "+locname[locpos])
		print ("============================================================")
		PATH = 'rawSAX/'+loct.upper()+'/'+col+loct+'.txt'
		rngm.getfile(PATH,graminput,byyear)
	elif (args["type"] == "corrosion"):
		PATH = 'sss/ef'+args["corrosion"]+'.txt'
		print(PATH)
		rngm.getfile(PATH,graminput,byyear,0)

