from tqdm import tqdm
import pandas as pd
import re
import nltk

email_df = pd.read_csv('test.csv')

#print(email_df.head())
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
def get_message(colln,idx):
    _msg = colln.iloc[idx]['message']
    msgs = _msg.split(':')[-1].split('/n')
    msgs_join = ' '.join(msgs)
    msg_sents = splitter.findall(msgs_join)
    #print('Dilip going to print -->',msg_sents)
    return msg_sents



# Saving actionable sentences in this list and then to a file.
sentences = []
tp = 0
fp = 0
fn = 0

pos_dict = {'MD':'1','VB':'2','VBG':'2','VBN':'2','VBP':'2','VBZ':'2','WP':'3'}
patterns = ['142','12','32','342','242']

def evaluate(idf):
    if idf[0]=='2':
        return True

    for p in patterns:
        if p in idf:
            return True
    return False

for k in tqdm(range(len(email_df))):
    try:
        sents_k = get_message(email_df,k)
        total_count+=len(sents_k)
        for s in sents_k:
            # Evaluation by findin patterns in nltk
            tags = nltk.pos_tag(s.strip().split())
            pos = [item[1] for item in tags]
            key_idf = ''
            for p in pos:
                if p in pos_dict.keys():
                    key_idf+=pos_dict[p]
                else:
                    key_idf+='4'
            #print("Dilip key_idf value",key_idf)
            gt_sent = evaluate(key_idf)
            #print('Dilip--> gt sent: ',gt_sent)
            if is_actionable(s.strip()):
                sentences.append(s.strip())
                #print('Dlip--< sentences :',sentences)
                if gt_sent:
                    tp+=1
                else:
                    fp+=1
            else:
                if gt_sent:
                    fn+=1
                    #print('Dilip tags :',tags)
        if len(sentences) > LIMIT:
            save_results = open('save_results.txt','a+')
            actionable_sentences+=len(sentences)
            print('Results Saved for ' + str(len(sentences)) + ' lines.')
            results = '\n'.join(sentences)
            save_results.write(results)
            save_results.close()
            sentences = []
    except:
        print('Message Not Retrieved')
#classify the email int he date set into Action/Dummy        
def detect(sen):
    for y in sentences:
        if y in sen:
            return "Action"
        
    return "Dummy"
email_df['State']=email_df['message'].apply(lambda x : detect(x))                                        
email_df.to_csv('email_classify.csv')
print('\n\n\n\n============== Summary ====================\n')
precission=(tp)/(tp+fp)
recall=(tp)/(tp+fn)
F1=2*precission*Recall/(precission+Recall)
print('Precision :', precission)
print('Recall :', recall)
print('F1 score : ',F1)
save_results = open('./outputs/save_results.txt','a+')
actionable_sentences+=len(sentences)
print('Results Saved for ' + str(len(sentences)) + ' lines.')
results = '\n'.join(sentences)
save_results.write(results)
save_results.close()


print('Number of Actionable Sentences : ', actionable_sentences)
print('Total Sentences : ', total_count)
