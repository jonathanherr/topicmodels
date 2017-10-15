#!/usr/bin/python
from gensim import corpora,models,utils
import sys
import os
if len(sys.argv)!=4:
	print "usage: tfidf.py <x>.dict (<x>.tfidf or <x>.tfidf.mm) datapath"
	sys.exit(0)
dictionary=corpora.Dictionary.load(sys.argv[1])
tfidf=None
tfidffile=sys.argv[2]
if tfidffile.endswith("tfidf.mm"):
	print "got corpus matrix, creating new tfidf model"
	corpus=corpora.MmCorpus(tfidffile)
	tfidf=models.TfidfModel(corpus,id2word=dictionary,normalize=True)
	tfidf.save(tfidffile.replace("tfidf.mm","tfidf"))
	print "saved tfidf model at "+ tfidffile.replace("tfidf.mm","tfidf")
else:
	print "using tfidf model at " + tfidffile
	tfidf=models.TfidfModel.load(tfidffile)
docs=[]
for dir,dirs,filenames in os.walk(sys.argv[3]):
	
	if not "/." in dir:
		for file in filenames:		
			path=dir+"/"+file
			if not path.startswith("."):
				doctfidf=tfidf[dictionary.doc2bow(utils.tokenize(utils.any2utf8(open(path,"r").read(),errors='ignore')))]
				doctfidf.sort(key=lambda tup:tup[1],reverse=True)
				docs.append((file,doctfidf))
tfidflog=open("tfidf.txt","w")
for file,doc in docs:
	for word in doc:
		tfidflog.write(file+","+tfidf.id2word[word[0]]+","+str(word[1])+"\n")	
tfidflog.close()
print "wrote results to tfidf.txt"	
#for term in tfidf.idfs:
#	print tfidf.id2word[term]+","+str(tfidf.idfs[term])
#	print tfidf[[(term,1)]]

