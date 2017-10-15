__author__ = 'jonathan'
from gensim import utils
from gensim import models
from gensim import corpora

import codecs
import os

cor=open("test.cor","w")
for dir,dirs,files in os.walk("testdata/corpora/chinese"):
    for file in files:
        utf8=True
        utf16=False
        lines=[]
        try:
            lines=codecs.open(dir+"/"+file,"r","utf-8").readlines()
            print "utf8"
        except:
            utf8=False
            pass
        if not utf8:
            print "utf16"
            try:
                lines=codecs.open(dir+"/"+file,"r","utf-16").readlines()
            except:
                utf8=False
                utf16=True
                pass

        for line in lines:
            cor.write(utils.any2utf8(line))
cor.close()
dictionary = corpora.Dictionary(utils.any2utf8(line.lower()).split() for line in codecs.open("testdata/corpora/chinese/BAXCN_00007421.txt","r", 'utf-16').readlines())
print "utf8 dictionary"
print dictionary.values()
dictionary.save("test.dict")
dictionary.save_as_text('dict.txt')
mydict=dictionary.load("test.dict")
print mydict
print dictionary



