from CDSDocument import CDSDocument
import pickle
import logging
import os

class CDS:
    """
    Operations on CDS documents.

    Reads index files and generates lists of CDSDocuments and provides access to documents.
    """

    def __init__(self, indexDoc, dataPath, datasetName):
        """
        init class

        indexDoc:locateion of neardupe index file
        dataPath:path to document corpora
        datasetName:name for this run - used to name some output files
        """
        self.index = indexDoc
        self.dataPath = dataPath
        self.datasetName = datasetName


    def readIndex(self):
        """
        Read index file and build CDSDocument for each document listed. Pickles resulting array of documents for future runs over the same index file.

        No Inputs
        Returns count of documents.
        """
        self.documents = {}
        lines = [line for line in open(self.index)]
        docCount = 0
        for line in lines:
            name, group, ispivot, similarity, parent, children, subgroup = range(7)
            fields = line.split(",")
            if len(fields[children]) > 0:
                childList = fields[children].split(";")
            else:
                childList = []
            doc = CDSDocument(fields[name], fields[group], fields[ispivot], fields[similarity], fields[parent], childList,
                fields[subgroup])
            self.documents[doc.name] = doc
            docCount += 1
        self.findDocuments() #locate all documents on disk
        self.pickleDocuments()
        return docCount


    def getDocuments(self):
        """return a map of documents, indexed by name(filename minus extension)"""
        return self.documents


    def getDocument(self, doc):
        """get document by key in documents list if it exists"""
        if doc in self.documents.keys():
            return self.documents[doc]
        else:
            return None


    def getPivots(self):
        """return array of pivots across the corpus"""
        pivotDocs = []
        for doc in self.documents:
            if self.documents[doc].isPivot:
                pivotDocs.append(self.documents[doc])
        return pivotDocs


    def findDocuments(self):
        """find a doc in the doc tree and add the full path to the document object"""
        for dirname, dirnames, filenames in os.walk(self.dataPath):
            for filename in filenames:
                shortName = filename[:len(filename) - 4] #remove .txt
                if shortName in self.documents.keys():
                    doc = self.documents[shortName]
                    doc.localPath = os.path.join(dirname, filename)
                    #print doc.localPath
                    logging.info("indexed document " + doc.localPath)
                else:
                    print "file " + filename + " not in index file."


    def pickleDocuments(self):
        """pickle the documents array"""
        doc_file = open(self.datasetName + "_documents.pkl", "wb")
        pickle.dump(self.documents, doc_file)
        doc_file.close()


    def unpickleDocuments(self):
        """open the pickle documents array. Assume name is <datasetname>._documents.pkl."""
        doc_file = open(self.datasetName + "_documents.pkl", "rb")
        self.documents = pickle.load(doc_file)


    def unpickleDocumentByName(self, pickleFileName):
        """open a pickled array of CDSDocuments. Open given name."""
        print pickleFileName
        doc_file = open(pickleFileName, "rb")
        self.documents = pickle.load(doc_file)
