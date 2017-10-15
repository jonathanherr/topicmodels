import ConfigParser
import os
import sys
from unittest import TestCase
import unittest
from LDA import LDA

__author__ = 'jonathan'

class TestLDA(TestCase):

    def setUp(self):
        self.config="testdata/simpletest.cfg"
        self.lda=LDA(configfile=self.config)
        documents=[]

        for filename in os.listdir(self.lda.datapath):
            if os.path.isfile(self.lda.datapath+"/"+filename):
                        documents.append(self.lda.datapath+"/"+filename)
        if os.path.exists(self.lda.corpusfile):
            os.remove(self.lda.corpusfile)
        self.lda.buildCorpus(documents)
        if os.path.exists(self.lda.modelfile):
            self.lda.loadModel()
        else:
            self.lda.generateModel()
    def test_initWithConfig(self):
        self.lda.initWithConfig(self.config)
        self.assertNotEqual(self.lda.datapath,"","config init failed")

    def test_initialize(self):
        config = ConfigParser.SafeConfigParser()
        config.read(self.config)
        self.lda.initialize(config)
    def test_convertCorpora(self):
        self.lda.initWithConfig(self.config)
        self.lda.convertCorpora(self.lda.corpusmatrixfile,"mm","blei")
        self.assertTrue(os.path.exists(self.lda.corpusmatrixfile.replace("mm","blei")),msg="conversion from mm to blei failed.")
        self.lda.convertCorpora(self.lda.corpusmatrixfile,"mm","low")
        self.assertTrue(os.path.exists(self.lda.corpusmatrixfile.replace("mm","low")),msg="conversion from mm to low failed.")

    def test_buildCorpus(self):
        self.lda.initWithConfig(self.config)
	#if os.path.exists(self.lda.corpusfile):
	#        os.remove(self.lda.corpusfile)
        documents=[]
	for filename in os.listdir(self.lda.datapath):
		if os.path.isfile(filename):
                	documents.append(self.lda.datapath+"/"+filename)
        self.lda.buildCorpus(documents)
        self.assertTrue(os.path.exists(self.lda.corpusfile),msg="Ran buildCorpus, but missing corpus file!")


    def test_inPhraseTable(self):
        ptable=open(self.lda.phrasetablefile,"r")
        for line in ptable.readlines():
            if self.lda.inPhraseTable(line):
                pass
            else:
                self.fail(msg="Phrase table line " + line + " failed test.")

    def test_matchRegex(self):
	import re
        self.lda.regexlist=[re.compile("(From:)")]
        teststring="From: test@test.com"
        self.assertTrue(self.lda.matchRegex(teststring),msg="Regex list:" + str(self.lda.regexlist) + " failed to match test string '" + teststring+"'")


    def test_preProcess(self):
        #pick a document to test from the datapath
        files=os.listdir(self.lda.datapath)
        doc=open(self.lda.datapath+"/"+files[0],"r").read()
        filtered_doc=self.lda.preProcess(self.lda.datapath+"/"+files[0])
        self.assertTrue(len(filtered_doc)<len(doc),msg="filtered document not shorter than original document.")


    def test_removeSpecials(self):
        specials="!@#$%^&*()_+=-~`';:/.,<>?\|"
        result=self.lda.removeSpecials(specials)
        self.assertTrue(len(result)==0,msg="Not all specials removed. Left with: " + result)


    def test_emailHeader(self):
        header="To: Someone@Somewhere.com"
        self.assertTrue(self.lda.emailHeader(header),msg="Email header ' " + header+"' not recognized.")

    def test_serializeCorpusMemoryFriendly(self):
	self.assertTrue(self.lda.corpusfile!=None,msg="self.lda.corpusfile is None!")
	if self.lda.dictionary is None:
		self.test_makeDictionaryMemoryFriendly()
        self.lda.serializeCorpusMemoryFriendly(self.lda.corpusfile,self.lda.dictionary)
        self.assertTrue(os.path.exists(self.lda.corpusmatrixfile),msg="Missing corpus matrix file")
        self.assertTrue(os.path.exists(self.lda.tfidfmodelfile),msg="Missing tfidf file.")

    def test_makeDictionaryMemoryFriendly(self):
        #todo: consider deleting old dictionary first
        self.lda.dictionary=self.lda.makeDictionaryMemoryFriendly(self.lda.corpusfile)
        self.assertTrue(os.path.exists(self.lda.dictionaryfile))


    def test_loadDictionary(self):
        self.lda.dictionary=self.lda.loadDictionary(self.lda.dictionaryfile)
        self.assertTrue(not self.lda.dictionary is None,msg="Failed to load dictionary")

    #@unittest.skip("not testing lsi models since we don't use them and they overwrite the lda model other functions rely on")
    #def test_generateLSIModel(self):
    #    self.lda.generateLSIModel(10)
    #    self.assertTrue(os.path.exists(self.lda.modelfile))

    def test_generateLDAModel(self):
        self.lda.generateLDAModel(10,1,False)
        self.assertTrue(os.path.exists(self.lda.modelfile))


    def test_printTopicList(self):
        self.lda.printTopicList(False)
        self.assertTrue(True)

    def test_writeTopicsFile(self):
	topics=self.lda.lda.show_topics(-1,int(self.lda.topnwords),True,True)
        self.lda.writeTopicsFile(topics,True)
        self.assertTrue(os.path.exists(self.lda.datasetName+".topics.weights.txt"),msg="Missing topic file '" + self.lda.datasetName+".topics.weights.txt'")
        self.lda.writeTopicsFile(topics,False)
        self.assertTrue(os.path.exists(self.lda.datasetName+".topics.txt"),msg="Missing topics file '" + self.lda.datasetName+".topics.txt'")

    def test_loadModel(self):
        self.lda.loadModel()
        if self.lda.modeltype=="LDA":
            self.assertTrue(self.lda.lda!=None,msg=self.lda.modeltype + " model failed to load.")
        elif self.lda.modeltype=="LSI":
            self.assertTrue(self.lda.lsi!=None,msg=self.lda.modeltype + " model failed to load.")
	else:
	    print "unknown modeltype " + self.lda.modeltype
    def test_documentTopics(self):
        topics=[]
	dir=0
        for file in os.listdir(self.lda.datapath):
	    if os.path.isfile(file):
            	docTopics=self.lda.getDocumentTopics(self.lda.datapath+"/"+file)
            	topics.append(docTopics)
	    else:
		dir+=1
        self.assertTrue(len(topics)==len(os.listdir(self.lda.datapath))-dir,msg="Didn't get topics for all the files in datapath '"+self.lda.datapath+"'")

    def test_removeNonAscii(self):
	#todo: read non-ascii string from a file - can't easily store in python file
        test="abc@#$123"
        valid="abc@#$123"
        result=self.lda.removeNonAscii(test)
        self.assertTrue(result==valid,msg="NonAscii character not removed.")

    def test_getDocText(self):
        text=""
        text=self.lda.getDocText(self.lda.datapath+"/"+os.listdir(self.lda.datapath)[0])
        self.assertTrue(len(text)>0,msg="doc text not returned.")

    def test_generateModel(self):
        self.lda.generateModel()
        self.assertTrue(os.path.exists(self.lda.modelfile))

    def test_convertDocumentsToCorpus(self):
        self.lda.convertDocumentsToCorpus()
        self.assertTrue(os.path.exists(self.lda.corpusfile),msg="corpus file '"+self.lda.corpusfile+"' not created!")
    def test_getTopWords(self):
	print "test_getTopWords"
        for file in os.listdir(self.lda.datapath):
	    print file
	    if os.path.isfile(self.lda.datapath+"/"+file):
                print "calling gettopwords on " + file
            	self.lda.getTopWords(file)
if __name__=="__main__":
	unittest.main()
