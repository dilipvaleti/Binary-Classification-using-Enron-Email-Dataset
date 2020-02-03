from tqdm import tqdm
import pandas as pd
import re
import nltk

df = pd.read_csv('emails.csv',nrows=1000)
print(df.head())
splitter = re.compile('[^.!?]*[\w]\s*[.!?]')
bow = {}
sentences = []
total_count = 0
actionable_sentences = 0;
LIMIT = 1000 #for reducing ram memory consumption

def is_actionable(query):
    # maintain a file and import common list action verbs
    # importing list of action words
    query = query.lower()
    words = ["?","can you ","would ","should ","please ","will ","could ","what ","what ","who "," how ","where ","when ",
            "do ","list ","document ","send ","forward ","fix ","write ","open ","wait",
            "move ","visit ","make ","listen ","come ","spend ","submit ","build ","bring ","ask ","grab",
            "read ","give ","act ","visit ","think ","drop ","call ","schedule ","accept ","reply ","respond",
            "create ","open ","close ","show ","make ","execute ","perform ","cause ","exercise ","practice",
            "answer ","serve ","manage ","arrange ","set ","name ","count ","get ","direct ","mail ","post",
            "transport ","ship ","commit ","charge ","institutionalize ","institutionalise ","air",
            "broadcast ","transmit ","repair ","restore ","secure ","determine ","define ","specify ","limit",
            "prepare ","ready ","deposit ","situate ","locate ","compose ","indite ","publish ","save",
            "spell ","spread ","unfold ","expect ","look ","await ","scan ","take ","study ","learn",
            "register ","show ","record ","translate ","understand "]
    for word in words:
        if word in query:
            if query.count(' ')>2:
                return True
    return False
