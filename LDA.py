#!/usr/bin/python
# coding:utf-8
import chardet
from gensim import corpora, models, utils
import configparser
import getopt
import logging
import os
import sys
import time
import re
import multiprocessing
from multiprocessing import Lock
import codecs

class MyCorpus:
    """
    Reads corpus from disk one line at a time to avoid loading the entire corpus into memory.
    Defines an iterator for accessing corpus one line at a time.
    """

    def __init__(self, corpusPath, dictionary):
        self.corpus = corpusPath
        self.dictionary = dictionary

    def __iter__(self):
        for line in open(self.corpus):
            yield self.dictionary.doc2bow(line.lower().split())


class LDA:
    """
    Interface with the gensim library to produce lda topic models.

    Relies on config file for setting various parameters. See config for documentation.
    Alternative execution via command line parameter may not be correct. TODO:fix command line parms or remove option.

    Common corpus building and inference procedures. Separate training functions for LDA, LSA and HDP topic modeling algorithms.

    """

    def __str__(self):
        from pprint import pprint

        x = str(pprint(vars(self)))
        return x

    def __init__(self, *args, **kwargs):
        """
        init with arbitary arguments so that we can have alternative constructors based on user input.

        If kwargs has key "configfile" then call initWthConfig else call initWithDataset

        """
        self.logpath="."
        self.datasetName="UNK"
        if "configfile" in kwargs:
            self.initWithConfig(kwargs.get("configfile"))
        else:
            self.initWithDataset(kwargs.get("datapath"), kwargs.get("datasetname"), kwargs.get("modeltype"))

    def initWithDataset(self, dataPath, datasetName, modeltype):
        """
        Initialize LDA with defaults and the datasetName given at the command line.
        Assumes these values can be set from the command line with falgs.
        """
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, filename="lda.log")
        self.modeltype = modeltype
        self.datapath = dataPath
        self.stopwordfile = "stopwords.txt"
        self.stoplist = [line.strip().lower().split(" ") for line in open(self.stopwordfile)][0]
        self.corpusfile = datasetName + ".cor"
        self.dictionaryfile = datasetName + ".dict"
        self.corpusmatrixfile = datasetName + ".mm"
        self.tfidfmatrixfile = datasetName + ".tfidf.mm"
        self.tfidfmodelfile = datasetName + ".tfidf"
        self.modelfile = datasetName + ".lda"
        self.datasetName = datasetName
        self.inferenceoutput = datasetName + ".docs.topics"
        self.topics = 25
        self.online = False
        self.numInferenceProcs = 4
        self.passes = 1
        self.maxdiff = 1
        self.minwordlength = 3
        self.topnwords = 20
        self.dictionary = None
        self.distributed = False
        self.logpath = "."

    def initWithConfig(self, configName):
        """
        read a config file, either the given one, or if that"s None, "lda.cfg"
        call initialize with the config object to init the class
        """
        config = configparser.SafeConfigParser()

        if configName is None:
            config.read("lda.cfg")
        else:
            if os.path.exists(configName):
                config.read(configName)
            else:
                print(("missing config file %s" % configName))
                logging.info("config file missing at " + configName)
                sys.exit()
        self.initialize(config)

    def initialize(self, config):
        """
        load config options into member vars.
        setup stoplist and phraselist and regexlist if turned on
        validate regex
        startup logging at <id>.lda.log

        requires a config object
        no return
        """
        try:

            if config.has_section("rules"):
                self.stopwordfile = config.get("rules", "stopwordfile")
                self.useStopWords = config.getboolean("rules", "usestopwordlist")
                self.usePhraseTable = config.getboolean("rules", "usephrasetable")
                self.useRegex = config.getboolean("rules", "useregex")
                self.regexfile = config.get("rules", "regexfile")
                self.phrasetablefile = config.get("rules", "phrasetablefile")
                self.minwordlength = config.getint("rules", "minwordlength")
                if self.useStopWords:
                    self.stoplist = [line.lower().strip().split(" ") for line in open(self.stopwordfile)][0]
                else:
                    self.stoplist = []
                if self.usePhraseTable:
                    self.phraselist = [line.strip().lower() for line in open(self.phrasetablefile)]
                else:
                    self.phraselist = []
                if self.useRegex:
                    self.regexlist = []
                    regexlist = [line.strip() for line in open(self.regexfile)]
                    for regex in regexlist:
                        try:
                            prog = re.compile(regex)
                            self.regexlist.append(prog)
                        except re.error:
                            print(("error in regex " + regex))
                            raise
                else:
                    self.regexlist = []
            self.logpath = config.get("general", "logpath")
            self.modeltype = config.get("general", "modeltype")
            self.lda = None
            self.corpus = None
            self.datapath = config.get("general", "data")
            self.datasetName = config.get("general", "id")
            self.numInferenceProcs = config.getint("general", "numInferenceProcesses")

            self.modelfile = config.get("files", "model")
            self.corpusfile = config.get("files", "corpus")
            self.dictionaryfile = config.get("files", "dictionary")
            self.corpusmatrixfile = config.get("files", "corpusmatrix")
            self.tfidfmatrixfile = config.get("files", "tfidfmatrix")
            self.tfidfmodelfile = config.get("files", "tfidfmodel")
            self.inferenceoutput = config.get("files", "inferenceoutput")

            self.dictionary = None  # must be preexisting for model generation
            if config.has_section("modelgeneration"):
                if config.has_option("modelgeneration", "topics"):
                    self.topics = config.getint("modelgeneration", "topics")
                else:
                    self.topics = 25
                if config.has_option("modelgeneration", "passes"):
                    self.passes = config.getint("modelgeneration", "passes")
                else:
                    self.passes = 1
                if config.has_option("modelgeneration", "maxdiff"):
                    self.maxdiff = config.getint("modelgeneration", "maxdiff")
                else:
                    self.maxdiff = 1
                if config.has_option("modelgeneration", "distributed"):
                    self.distributed = config.getboolean("modelgeneration", "distributed")
                else:
                    self.distributed = False
                if config.has_option("modelgeneration", "topnwords"):
                    self.topnwords = config.get("modelgeneration", "topnwords")
                else:
                    self.topnwords = 20
                if config.has_option("modelgeneration", "online"):
                    print(("online=" + config.get("modelgeneration", "online")))
                    self.online = config.getboolean("modelgeneration", "online")
                    print((self.online))
                else:
                    self.online = False
                if config.has_option("modelgeneration", "update_every"):
                    self.update_every = config.get("modelgeneration", "update_every")
                else:
                    self.update_every = 1
                if config.has_option("modelgeneration", "chunksize"):
                    self.chunksize = config.get("modelgeneration", "chunksize")
                else:
                    self.chunksize = 10000
                if config.has_option("modelgeneration", "dictionary_size"):
                    self.dict_size = config.getint("modelgeneration", "dictionary_size")
                if config.has_option("modelgeneration", "no_above"):
                    self.no_above = config.getfloat("modelgeneration", "no_above")
                if config.has_option("modelgeneration", "no_below"):
                    self.no_below = config.getfloat("modelgeneration", "no_below")

        except configparser.NoOptionError as err:
            print((str(err)))
        except configparser.NoSectionError as err:
            print((str(err)))
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO,
                            filename=self.logpath + os.sep + self.datasetName + ".lda.log")

    def convertCorpora(self, path, currentFormat, newFormat):
        """
        convert a corpus from one format to another
        supports mm, blei and low formats, both directions.
        """
        corpus = None
        print(("convert " + path + " from " + currentFormat + " to " + newFormat))
        if currentFormat == "mm":
            corpus = corpora.MmCorpus(path)
        elif currentFormat == "blei":
            corpus = corpora.BleiCorpus(path)
        elif currentFormat == "low":
            corpus = corpora.LowCorpus(path)
        dictionary = self.loadDictionary(self.dictionaryfile)

        if newFormat == "mm":
            corpora.MmCorpus.serialize(path.replace(currentFormat, newFormat), corpus, id2word=dictionary)
        elif newFormat == "blei":
            corpora.BleiCorpus.serialize(path.replace(currentFormat, newFormat), corpus, id2word=dictionary)
        elif newFormat == "low":
            corpora.LowCorpus.serialize(path.replace(currentFormat, newFormat), corpus, id2word=dictionary)

    def buildCorpus(self, documents=None, lock=None):
        """
        Build a document with one document per line to condence corpus into single file
        some filtering done here - only alpha words included, no email headers, and no special characters included. URLs and email addresses also removed.
        most filtering done at dictionary build time.
        here we only remove enough to eliminate documents which are empty of content(for example, forwarded emails with no body but an attachment, or spreadsheets of all numbers)
        """
        logging.info("building corpus file " + self.corpusfile)
        count = 0.0
        lastPer = 0
        if documents is None:
            docs = []
        else:
            docs = documents
        # logging.info("start buildCorpus")
        # logging.info("reading corpora at " + self.datapath)
        # if os.path.exists(self.corpusfile):
        #    os.remove(self.corpusfile)
        if documents is None:
            for dirname, dirnames, filenames in os.walk(self.datapath):
                for filename in filenames:
                    docs.append(os.path.join(dirname, filename))
        fileCount = len(docs)
        print(("%s files in corpus." % str(fileCount)))
        for path in docs:
            print(path)
            count += 1
            if round(count / fileCount * 100) % 5 == 0:
                per = round(count / fileCount * 100)
                if per > lastPer:
                    print(("%s: %s  complete" % (multiprocessing.current_process().name, str(per) + "%")))
                    lastPer = per

            doc, enc = self.preProcess(path)
            if len(doc.strip()) > 0:
                if not lock is None:
                    lock.acquire()
                print(("writing corpus with encoding %s" % enc))

                # corpus = codecs.open(self.corpusfile.replace(".cor","."+enc+".cor"), mode="a",encoding=enc)
                corpus = codecs.open(self.corpusfile, mode="a", encoding="UTF8")
                # udoc=unicode(doc,encoding=enc)
                # udoc=doc.decode(encoding=enc,errors="ignore")
                corpus.write(doc + " \n ")
                corpus.close()
                if not lock is None:
                    lock.release()
            else:
                logging.warning(path + " has no content to write to corpus file")
        logging.info("built corpus of " + str(count) + " documents")
        return fileCount

    def inPhraseTable(self, line):
        """
        Test if a phrase from the phraselist file is in a line. lowercase line. log to debug when found.
        """
        for phrase in self.phraselist:
            if phrase in line.lower():
                logging.debug("skipped line\n" + line + "\nb/c found phrase " + phrase)
                return True
        return False

    def matchRegex(self, line):
        """
        Test if a line matches any regex in the regexlist
        """
        for regex in self.regexlist:
            if not regex.search(line) is None:
                logging.debug("regex detected match in line " + line)
                return True
        return False

    def guessFileEncoding(self, docPath):
        """
        Use CharDet to guess the file encoding. Take whatever it suggests and log the likelihood info. Replace "ascii" with "UTF8"
        """
        encGuess = chardet.detect(open(docPath).read())
        print(encGuess)
        logging.info("chardet says %s has encoding:%s" % (docPath, str(encGuess)))
        encoding = encGuess["encoding"]
        if encoding == "ascii":
            encoding = "UTF-8"
        if encoding is None:
            encoding = "UTF-8"
        return encoding

    def preProcess(self, docPath):
        """
        Main function for removing junk from input text, both during training and inference and corpus building.

        Removes nonascii chars, lowercases, removes email headers(any string like From:/To: etc - any string where colon precdes a space
        Removes regex based on regex list.
        Skips all URLs and email addresses and words below set minwordlength
        removes all special characters
        """
        doc = ""
        encoding = "utf-8"
        if os.path.isfile(docPath):
            #encoding = self.guessFileEncoding(docPath)
            docFile = codecs.open(docPath, encoding=encoding).readlines()
            lines = [line.strip().lower().replace("\t"," ").replace("\n"," ").replace("\\n"," ").replace("\\t"," ") for line in docFile if os.path.isfile(docPath)]

            for line in lines:
                if not self.emailHeader(line) and not self.matchRegex(line) and not self.inPhraseTable(line):
                    if encoding == "utf-8":
                        for word in line.split(" "):
                            if len(word) >= int(self.minwordlength):
                                word_nospecials = self.removeSpecials(word.lower())
                                if len(word_nospecials) >= int(self.minwordlength):
                                    doc += word_nospecials + " "
                    else:
                        for word in line.split(" "):
                            word_nospecials = self.removeSpecials(str(word), False)
                            if len(word_nospecials) >= 0:
                                doc += word_nospecials + " "
        return doc, encoding

    def removeSpecials(self, word, alphaonly=False):
        """
        Removes any character that is not alphabetic or space
        """
        symbols = "!@#$%^&*()[]{}-_=+;:?><,./"
        symbols=""
        return "".join(e for e in word if e not in symbols and (e.isalpha() or not alphaonly))

    # return "".join(e for e in word if e.isalpha() or e.isspace())

    def emailHeader(self, line):
        """
        return true if email header field - except subject, which we leave in for its content
        """
        if line.find(" ") < line.find(":") or line.find(":") < 0:
            if line.find("X-") < 0 or "subject:" in line.lower():
                return False
        return True

    def serializeCorpusMemoryFriendly(self, corporaFilePath, dictionary):
        """
        serialize corpus to disk while loading only one document at a time into memory
        """
        if dictionary is None:
            print("Dictionary cannot be none when serializing corpus")
            raise TypeError
        if corporaFilePath is None:
            print("Corpora file path cannot be None!")
            raise TypeError
        # if not os.path.exists(corporaFilePath):
        #    print "corpora file at '" + corporaFilePath + "' is missing."
        #    raise TypeError

        logging.info("start serializing corpus to vector space")
        corpus = MyCorpus(corporaFilePath, dictionary)
        tfidf = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
        tfidf.save(self.tfidfmodelfile)
        print("writing tfidf model")
        corpora.MmCorpus.serialize(self.tfidfmatrixfile, tfidf[corpus], progress_cnt=10000)
        print("writing bow model")
        corpora.MmCorpus.serialize(self.corpusmatrixfile, corpus, progress_cnt=10000)
        logging.info("end serialize corpus")

    def makeDictionaryMemoryFriendly(self, corporaFilePath):
        """
        create a dictionary in a memory friendly way without ever loading all documents into memory
        """
        logging.info("start make dictionary")
        print("making dictionary")
        print(("dictionary size:" + str(self.dict_size)))
        print(("no_above:" + str(self.no_above)))
        print(("no_below:" + str(self.no_below)))
        cor = []
        """for dir,dirs,filenames in os.walk(os.path.dirname(corporaFilePath)):
            for file in filenames:
                if file.endswith(".cor"):
                    enc=file.split(".")[len(file.split("."))-2]
                    cor.extend(line.lower().split() for line in codecs.open(os.path.join(dir,file),encoding=enc))
        """
        cor = [line.lower().split() for line in codecs.open(self.corpusfile, encoding="UTF8")]
        dictionary = corpora.Dictionary(cor)
        print(dictionary)
        stop_ids = [dictionary.token2id[stopword] for stopword in self.stoplist if stopword in dictionary.token2id]
        dictionary.filter_tokens(stop_ids)  # remove stop words
        dictionary.filter_extremes()
        dictionary.compactify()  # remove gaps in id sequence after words that were removed
        dictionary.save(self.dictionaryfile)
        dictionary.save_as_text(self.datasetName + "_wordids.txt")
        del dictionary  # delete this copy and reload to save memory
        dictionary = corpora.Dictionary.load(self.dictionaryfile)

        logging.info("end make dictionary")
        return dictionary

    def loadDictionary(self, path):
        """
        Read a dictionary from disk
        """
        return corpora.Dictionary.load(path)

    def generateHDPModel(self):
        """
        Create an HDP model and print its topics to the log
        """
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary.load(self.dictionaryfile)
            self.corpus = corpora.MmCorpus(self.corpusmatrixfile)

        hdp = models.HdpModel(self.corpus, self.dictionary)
        hdp.print_topics(self.topics, self.topnwords)

        hdp.save(self.datasetName + ".hdp")

    def generateLSIModel(self, topics):
        """
        Create an LSI model and print its topics to the log
        """
        dictionary = corpora.Dictionary.load(self.dictionaryfile)
        corpus = corpora.MmCorpus(self.corpusmatrixfile)
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topics)
        lsi.print_topics(-1)
        lsi.save(self.datasetName + ".lsi")

    def generateLDAModel(self, topics, passes, distribute):
        """
        generate an LDA model. Use configured dictionary and corpus filenames
        """
        print(("topics=" + str(topics) + "\npasses=" + str(passes) + "\ndistributed=" + str(distribute)))
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary.load(self.dictionaryfile)
        if self.corpus is None:
            self.corpus = corpora.MmCorpus(self.tfidfmatrixfile)
        logging.info(self.corpus)
        if self.online:
            lda = models.LdaModel(self.corpus, id2word=self.dictionary, num_topics=topics, passes=1,
                                  update_every=int(self.update_every), chunksize=int(self.chunksize),
                                  distributed=distribute)  # online lda
        else:
            lda = models.LdaModel(self.corpus, id2word=self.dictionary, num_topics=topics, passes=passes,
                                  update_every=0, distributed=distribute)  # batch process, no updates
        lda.save(self.modelfile)

        topics = lda.show_topics(-1, int(self.topnwords), True,
                                 True)  # all topics, topnwords words each, write to log, write nice formatting
        self.writeTopicsFile(topics, False)
        self.writeTopicsFile(topics, True)

    def printTopicList(self, includeweights):
        """
        load model and print topic list
        """
        if os.path.exists(self.modelfile):
            print("loading model")
            lda = models.LdaModel.load(self.modelfile)
            topics = lda.show_topics(-1, int(self.topnwords), True, True)
            self.writeTopicsFile(topics, includeweights)
        else:
            print("missing lda model file")

    def writeTopicsFile(self, topics, includeweights):
        """
        write given topic list to file
        """
        fileName = self.datasetName + ".topics.txt"
        if includeweights:
            fileName = self.datasetName + ".topics.weights.txt"
        topicfile = codecs.open(fileName, "w", encoding="UTF8")
        x = 1
        for topic in topics:
            topic=topic[1] #topic[0] is the topic number
            if not includeweights:
                words = self.removeWeightsTopicWordList(topic)
                topicfile.write(str(x) + ":" + ",".join(words) + "\n")
            else:
                topicfile.write(str(x) + ":" + topic + "\n")
            x += 1
        topicfile.close()

    def loadModel(self):
        """
        load the lda or lsi model into memory. store topic list. load dictionary and corpus matrix from disk
        """
        print(("loading " + self.modeltype + " model " + self.modelfile))
        if self.modeltype == "LDA":
            if os.path.exists(self.modelfile):
                self.lda = models.LdaModel.load(self.modelfile)
                self.alltopics = self.lda.show_topics(-1, 20, True, True)
                self.dictionary = corpora.Dictionary.load(self.dictionaryfile)
                self.corpus = corpora.MmCorpus(self.corpusmatrixfile)
            # self.lda.print_topics(-1)
            else:
                print(("missing " + self.modelfile + " file. Please train a model before running inference.\n"))
                raise Exception()
        elif self.modeltype == "LSI":
            self.lsi = models.LsiModel.load(self.modelfile)
            self.alltopics = self.lsi.show_topics(-1, 10, True, True)
            self.dictionary = corpora.Dictionary.load(self.dictionaryfile)
            self.corpus = corpora.MmCorpus(self.corpusmatrixfile)
            self.lsi.print_topics(-1)

    def getDocumentTopics(self, docPath):
        """
        perform inference on a document against the loaded model
        """
        if len(docPath) > 0:
            logging.info("reading document " + docPath)
            start = time.clock()
            text = self.getDocText(docPath)
            topics = []
            doc_model = None
            if self.modeltype == "LDA":
                if self.lda is None:
                    raise Exception(
                        "Missing LDA object. Build model first and load model before calling getDocumentTopics.")
                doc_model = self.lda[self.dictionary.doc2bow(text)]
            elif self.modeltype == "LSI":
                print((self.dictionary.doc2bow(text)))
                doctfidfmodel = models.TfidfModel(self.dictionary.doc2bow(text), normalize=True)
                doc_model = self.lsi[doctfidfmodel]

            sorted_topics = sorted(doc_model, key=lambda topic: topic[1], reverse=True)
            for topicnum, weight in sorted_topics:
                t = self.alltopics[topicnum]
                words = self.removeWeightsTopicWordList(t)
                topics.append("|".join(words) + ":" + str("%.3f" % round(float(weight), 5)))
            end = time.clock()
            logging.info(docPath + "\n%.2gs" % (end - start) + "\n\n")
            return topics
        else:
            return []

    def inferenceOnList(self, documents, lock):
        """
        documents: list of paths to documents to be inferenced
        """
        count = 0
        lastPer = 0

        print(("Process " + multiprocessing.current_process().name + " working on " + str(len(documents))))
        for document in documents:
            # print "writing inference for document " + document + " to " + lda.inferenceoutput
            count += 1
            if round(count / len(documents)) % 5 == 0:
                per = round(count / len(documents))
                if per > lastPer:
                    print((str(per) + "% complete=" + multiprocessing.current_process().name))
                    lastPer = per

            topics = lda.getDocumentTopics(document)
            logging.info(topics)
            doc = CDSDocument(os.path.basename(document), "", True, 0, "", "", "")
            doc.topics = topics
            # doc.topWords=getTopWords(document)
            lock.acquire()
            out = open(lda.inferenceoutput, "a")
            out.write(doc.getDetails() + "\n")
            out.close()
            lock.release()

    def getTopWords(self, document):
        """
        use the TFIDF model to get the top words in the document
        """
        print("gettopwords")
        text = open(self.datapath + "/" + document, "r").read()
        dictionary = self.loadDictionary(self.dictionaryfile)
        tfidfmodel = models.TfidfModel.load(self.tfidfmodelfile)
        print(dictionary)
        print((dictionary.doc2bow(text)))
        print((tfidfmodel[dictionary.doc2bow(text)]))

    def doInference(self, documents):
        """
        Start multiple processes running the inferenceOnList method, splitting the workload evenly among them.
        """
        lock = Lock()
        numDocs = len(documents)
        end = -1
        numProcs = int(self.numInferenceProcs)
        jobs = []
        print(("performing inference on " + str(len(documents)) + " documents"))
        for i in range(numProcs):
            start = end + 1
            end = start + (numDocs / numProcs)
            if i == numProcs - 1 and end < numDocs and numDocs - end < (numDocs / numProcs):
                end = numDocs
            print(("sending document range " + str(start) + ":" + str(end) + " to new proc"))

            p = multiprocessing.Process(target=self.inferenceOnList, args=(documents[start:end], lock))
            jobs.append(p)
            p.start()

    def removeWeightsTopicWordList(self, topic):
        """
        strip off the word weight in topic word list formatted like "word*.01+word2*.02"
        """
        terms = topic.split("+")
        words = []
        for term in terms:
            word = term.split("*")[1]
            words.append(word)
        return words

    def removeNonAscii(self, s):
        """
        Remove non-ascii characters
        """
        return "".join(i for i in s if ord(i) < 128)

    def getDocText(self, docPath):
        """
        Filter document for junk and format text in document into array of tokens. use gensim.utils.tokenize
        TODO: add lemmatization
        """
        document, encoding = self.preProcess(docPath)

        texts = [word for word in utils.tokenize(document) if word not in self.stoplist]
        return texts

    def generateModel(self):
        """
        Generate a model(LDA,HDP,LSI) based on settings already set in the object.
        Report time to generate model.
        """
        global startLoadModel, doneLoadModel
        print(("model type:" + self.modeltype))
        startLoadModel = time.time()
        if not os.path.exists(self.dictionaryfile):
            print(("creating dictionary " + self.dictionaryfile))
            self.dictionary = self.makeDictionaryMemoryFriendly(self.corpusfile)
        else:
            print(("reading dictionary " + self.dictionaryfile))
            self.dictionary = self.loadDictionary(self.dictionaryfile)
        if not os.path.exists(self.corpusmatrixfile):
            print(("serializing corpus " + self.corpusmatrixfile))
            self.serializeCorpusMemoryFriendly(self.corpusfile, self.dictionary)
        if self.modeltype == "LDA":
            print(("creating lda model with " + str(self.topics) + " topics."))
            if self.online:
                print(("Running in online mode with chunksize: " + str(self.chunksize)))
            else:
                print(("Running in offline mode with " + str(self.passes) + " passes"))
            if self.distributed:
                print("Running in distributed mode")
            self.generateLDAModel(self.topics, self.passes, self.distributed)
        elif self.modeltype == "LSI":
            print("creating lsi model")
            self.generateLSIModel(topics)
        elif self.modeltype == "HDP":
            print("creating hdp model")
            self.generateHDPModel()
        doneLoadModel = time.time()
        print(("Time to generate model:" + str(doneLoadModel - startLoadModel)))

    def convertDocumentsToCorpus(self):
        """
        Read documents from self.datapath and convert to a single large corpus file. Use the buildCorpus method, run over numProcs procs
        """
        global jobs, lock, documents, dirpath, dirs, filenames, filename, numDocs, start, end, numProcs, corpus, i, p
        print("building corpora")
        jobs = []
        lock = Lock()
        documents = []
        for dirpath, dirs, filenames in os.walk(self.datapath):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    documents.append(filepath)
        numDocs = len(documents)
        start = 0
        end = -1
        numProcs = int(self.numInferenceProcs)
        # clear old corpus file before we make a new one
        corpus = open(self.corpusfile, "w")
        corpus.write("")
        corpus.close()
        for i in range(numProcs):
            start = end + 1
            end = start + int(numDocs / numProcs)
            if i == numProcs - 1 and end < numDocs and numDocs - end < (numDocs / numProcs):
                end = numDocs
            print(("sending document range " + str(start) + ":" + str(end) + " to process for buildCorpus"))
            p = multiprocessing.Process(target=self.buildCorpus, args=(documents[start:end], lock))
            p.start()

    def inferTopicsFromIndexFile(self, pivotFile):
        """
        assumes use of a neardupe index file (aka pivotfile).
        note - this method reqries a <datasetname>_documents.pkl file which is a list of objects representing each document in the corpus that is to be inferenced, including its path on disk and details from the neardup index file.
        """
        cds = CDS(pivotFile, self.datapath, self.datasetName)

        # if we"ve already generated a set of CDS document objects, use that, otherwise generate a new one. Can be time consuming, so first runs will take longer than subsequent runs.
        if not os.path.exists(self.datasetName + "_documents.pkl"):
            print((
                  "***   Missing document index at " + self.datasetName + "_documents.pkl. Creating the index is very slow and may take an hour or more. ***"))
            cds.readIndex()
        else:
            print(("Reading document details from " + self.datasetName + "_documents.pkl. This may take a moment."))
            cds.unpickleDocuments()

        pivots = cds.getPivots()
        docs = cds.getDocuments()
        docList = open("doclist.txt", "w")
        docList.write(str(docs))
        docList.close()
        self.loadModel()
        pivotCount = len(pivots)

        print("sending corpora to pool")
        jobs = []
        workers = 4
        endRange = 0
        lock = Lock()
        for i in range(workers):
            print(i)
            startRange = endRange
            endRange = startRange + len(pivots) / workers
            print(startRange)
            print(endRange)
            # add any remainder
            if len(pivots) - endRange < (len(pivots) / workers):
                endRange = len(pivots)
            p = multiprocessing.Process(target=self.getPivotInference, args=(cds, pivots[startRange:endRange], lock))
            jobs.append(p)
            p.start()
        # pool.map(getPivotInference,[(pivot,cds) for pivot in pivots])
        print(("pivots:" + str(pivotCount)))

        logging.info(str(pivotCount) + " pivots in index")

    def getPivotInference(self, cds, pivots, lock):
        """
        perform inference on the given pivot document and its children
        """
        print((multiprocessing.current_process().name))
        print(("processing " + str(len(pivots)) + " pivots."))

        for pivot in pivots:
            print((multiprocessing.current_process().name + ":pivot=" + pivot.localPath))
            pivot.topics = self.getDocumentTopics(pivot.localPath)
            for child in pivot.children:
                print((multiprocessing.current_process().name + "child=:" + child))
                childDoc = cds.getDocument(child)

                if not childDoc is None and child != "" and float(childDoc.similarity) < 1.0:
                    childDoc.topics = self.getDocumentTopics(childDoc.localPath)
                elif childDoc is None:
                    print(("couldnt find " + child))
                elif child != "" and float(childDoc.similarity) == 1.0:
                    childDoc.topics = pivot.topics
        lock.acquire()
        topicLog = open(self.inferenceoutput, "a")
        for pivot in pivots:
            topicLog.write(pivot.getDetails() + "\n")
            for child in pivot.children:
                childDoc = cds.getDocument(child)
                if not childDoc is None:
                    topicLog.write(childDoc.getDetails() + "\n")
        topicLog.close()
        lock.release()


def usage():
    """
    Usage example
    """
    print(
        "LDA.py \n -c --config \t\t <config file> If used, flags to indicate command can be supplied instead of full datasetname, corpora and command option parms. Note that required parameters must exist in the config file. \n\t\t\t Can be used with --inference, --buildcorpora and --generatemodel. For instance . / LDA.py - cdemo.cfg - -buildcorpora will build corpora using settings in config file. \n -n --name \t\t <name of dataset> Required. Used to name output files and indicate names of input files. \n -d --corpora\t\t <path to documents>  Required for g and b\n -g --generatemodel \t generate a model from the given corpora \n -r --distributed \t used with -g to distribute to running workers. see README \n -i --inference \t <path to index file> Provide the name of index file to perform inference over \n -b --buildcorpora \t perform corpora building pre-process step \n -m --modeltype \t <LDA|LSI> Indicate model type. Must be LDA, LSI or HDP. Required with -g and -i. \n -u --document \t\t single document, or comma separated list of documents to run inference on. \n -f --infolder \t\t <path to input folder> Alternative to providing index file. Run inference on documents in given folder. \n -t --topics \t\t <int> Optionally indicate the number of topics to generate(for lda model generation), defaulting to 10. \n -p --passes \t\t <int> Set the number of passes to use during lda model generation, defaulting to 1. \n -w --writetopiclist \t Write out topic list from datasets model to topics.txt \n\n Examples: \nBuild Corpora\n. / LDA.py - -name = demo - -corpora =../ corpora / Text / 001 - b - -modeltype = LDA \nor\n. / LDA.py - c demo.cfg - -buildcorpora \n\nGenerate Model\n. / LDA.py - -name = demo - -corpora =../ corpora / Text / 001 - g - -modeltype = LDA - -topics = 25 - -passes = 10 \nor\n. / LDA.py - c demo.cfg - g \n\n Inference\n. / LDA.py - -name = demo - -inference = bindex.txt - -modeltype = LDA \nor\n. / LDA.py - c demo.cfg - -inference = index.txt\n")


if __name__ == "__main__":
    corpus = ""
    dataPath = ""
    datasetName = ""
    configFile = ""
    enerateModel = False
    modeltype = ""
    buildCorpora = False
    docs = False
    documents = []
    lda = None
    topics = 25
    passes = 1
    pivotFile = None
    inferenceout = None
    getTopics = False
    distributed = False
    corporaToConvert = ""
    opts, remainder = getopt.getopt(sys.argv[1:], "c:n:dbt:gm:u:i:v:p:f:wro:",
                                    ["config=", "name=", "corpora=", "buildcorpora", "topics=", "generatemodel",
                                     "modeltype=", "document=",
                                     "inference=", "convert=", "passes=", "infolder=", "writetopiclist", "distributed",
                                     "inferenceout"])
    convert = False
    writeTopics = False
    readConfig = False
    generateModel = False
    if len(opts) < 1:
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-n", "--name"):
            corpus = a
            datasetName = a
        elif o in ("-c", "--config"):
            readConfig = True
            configFile = a
        elif o in ("-b", "--buildcorpora"):
            buildCorpora = True
        elif o in ("-g", "--generatemodel"):
            generateModel = True
        elif o in ("-u", "--document"):
            docs = True
            documents = a.split(",")
        elif o in ("-f", "--infolder"):
            docs = True
            for dirpath, dirs, filenames in os.walk(a):
                for filename in filenames:
                    if os.path.isfile(os.path.join(dirpath, filename)):
                        documents.append(os.path.join(dirpath, filename))
        elif o in ("-m", "--modeltype"):
            modeltype = a
        elif o in ("-t", "--topics"):
            topics = int(a)
        elif o in ("-p", "--passes"):
            passes = int(a)
        elif o in ("-i", "--inference"):
            getTopics = True
            pivotFile = a
        elif o in ("-o", "--inferenceout"):
            inferenceout = a
        elif o in ("-d", "--corpora"):
            dataPath = a
        elif o in ("-v", "--convert"):
            corporaToConvert = a
            convert = True
        elif o in ("-w", "--writetopiclist"):
            writeTopics = True
        elif o in ("-r", "--distributed"):
            distributed = True

    if datasetName != "" or readConfig:
        if readConfig:
            lda = LDA(configfile=configFile)
        else:
            lda = LDA(datapath=dataPath, datasetname=datasetName, modeltype=modeltype)
            lda.passes = passes
            lda.topics = topics
    else:
        usage()
        sys.exit(2)

    if getTopics:
        if pivotFile == "" and not readConfig:
            usage()
            print("-i=<pivotFile> is required")
            sys.exit(2)
        if not inferenceout is None:
            lda.inferenceoutput = inferenceout

        print(("writing inference to " + lda.inferenceoutput))

        lda.inferTopicsFromIndexFile(pivotFile)
    elif convert:
        print(("converting " + corporaToConvert + " to blei format"))
        lda.convertCorpora(corporaToConvert, "mm", "blei")
    elif writeTopics:
        lda.printTopicList(True)  # write with word weights
        lda.printTopicList(False)  # write without word weights
    elif buildCorpora:
        lda.convertDocumentsToCorpus()
    elif generateModel:
        if os.path.exists(lda.corpusfile):
            lda.generateModel()
        else:
            print("generate corpora before generating model")
            usage()
    elif docs:
        lda.loadModel()
        out = open(lda.inferenceoutput, "w")
        out.write("")
        out.close()
        lda.doInference(documents)
