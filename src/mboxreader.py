import mailbox
import gzip
import os
import codecs

def getbody(message): #getting plain text 'email body'
    body = None
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    elif message.get_content_type() == 'text/plain':
        body = message.get_payload(decode=True)
    return body
def clean(text):
    if text is None:
        text=""
    names=["jonathan","allie""eli","ari","herr"]
    text=text.lower()
    text=text.replace("\r\n"," ").replace("\n"," ").replace("\t"," ").replace("\\n"," ").replace("\\t"," ").strip()
    for name in names:
        if name in text:
            text=text.replace(name," ")
    return text
mboxchunk_filename="chunk_"
msgnum=0
max=50000
out=codecs.open("mail.csv",mode="w",encoding="utf-8",errors="ignore")

for dirpath, dirnames, filenames in os.walk("graymail"):
    for file in filenames:
        if file.startswith(mboxchunk_filename):
            mbox = mailbox.mbox(os.path.join(dirpath,file))

            for message in mbox:
                body=getbody(message)
                if body is None:
                    print("no body!")
                    body=""
                if len(body)>0:
                    body=clean(str(body))

                out.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(message["Message-ID"],message["From"],message["To"],message["Date"],clean(message["subject"]),body))
                if msgnum<max:
                    print(message['subject'])
                    msgnum+=1
                else:
                    print("hit message max of %i"%max)
                    exit(0)
out.close()