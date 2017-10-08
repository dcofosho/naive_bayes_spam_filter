import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

df = pd.read_table('SMSSpamCollection',
		sep='\t',
		header=None,
		names=['label', 'sms_message'])
#print('head:'+str(df.head()))

df['label'] = df.label.map({'ham':0, 'spam':1})
#print('shape: '+str(df.shape))
#print('head2: '+str(df.head()))

msgList = list(df.sms_message)
#print("msgList: "+str(msgList))

count_vector.fit(msgList)
word_list = count_vector.get_feature_names()

#print("FEATURE NAMES \n"+str(word_list))

msg_array = count_vector.transform(msgList).toarray()
#print("MSG ARRAY \n"+str(msg_array))

freq_matrix = pd.DataFrame(msg_array, columns = word_list)
print("FREQ MATRIX \n"+str(freq_matrix))
