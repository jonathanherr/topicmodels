'''
Read a path and an index file, remove everything from the index file not in the path and write a new index file
'''
import os
import sys
from LDA import CDS
from LDA import CDSDocument

if len(sys.argv)==4:
	startPath=sys.argv[1]
	indexFile=sys.argv[2]
	outputFile=sys.argv[3]
else:
	print "createIndexFile.py startPath indexFile outputFile"
	sys.exit(1)

index=open(indexFile,"r").read()
rows=index.split("\n")
filelist=[]
for dir,dirs,files in os.walk(startPath):
	filelist.extend(files)

cds=CDS(indexFile,startPath,"test")
print "reading index file"
cds.unpickleDocumentsByName(indexFile)


newindex=open(outputFile,"w")
for filename in filelist:
	shortName=filename[0:len(filename)-4]
	print shortName
	doc=cds.getDocument(shortName)
	print doc.getDetails()
	newindex.write(doc.getDetails()+"\n")
newindex.close()
print "wrote file " + outputFile



