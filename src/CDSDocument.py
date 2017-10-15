class CDSDocument:
    """
    Represent a document from a CDS index
    Call getDetails to return a comma separated line with all known document details, including topics.
    """
    def __init__(self, name, group, isPivot, similarity, parent, childList, subgroup):
        """
        Populate document properties
        """
        self.name = name
        self.group = group
        if isPivot == "Pivot":
            self.isPivot = True
        else:
            self.isPivot = False
        self.similarity = similarity
        self.parent = parent
        self.children = childList
        self.subgroup = subgroup
        self.localPath = ""
        self.topics = []

    def __print__(self):
        return self.printDoc()

    def __repr__(self):
        return self.getDetails()

    def printDoc(self):
        """
        Print to std out the known properties of the document
        """
        print self.getDetails() + "\n"

    def getDetails(self):
        """
        Return the known details of the document
        """
        return self.name + "," + self.group + "," + str(self.isPivot) + "," + str(self.similarity) + "," + ";".join(
            self.children) + "," + self.subgroup.strip() + "," + ";".join(self.topics)