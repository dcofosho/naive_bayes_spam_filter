import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
count_vector = CountVectorizer()

#Read labeled sms data with pandas as table with columns label and sms message
df = pd.read_table('SMSSpamCollection',
		sep='\t',
		header=None,
		names=['label', 'sms_message'])
#print('head:'+str(df.head()))

#Replace the "ham" (non-spam) label with 0 and the spam label with 1
df['label'] = df.label.map({'ham':0, 'spam':1})
#print('shape: '+str(df.shape))
#print('head2: '+str(df.head()))

#Get messages as list
msgList = list(df.sms_message)
#print("msgList: "+str(msgList))

#Fit count_vector to list and get word list
count_vector.fit(msgList)
word_list = count_vector.get_feature_names()

#print("FEATURE NAMES \n"+str(word_list))
#convert list to frequency array
msg_array = count_vector.transform(msgList).toarray()
#print("MSG ARRAY \n"+str(msg_array))

#convert frequency array to freq matrix
freq_matrix = pd.DataFrame(msg_array, columns = word_list)
#print("FREQ MATRIX \n"+str(freq_matrix))

#Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
						df['label'],
						random_state=1)
print('Total num rows:\n {}'.format(df.shape[0]))
print('Num rows in training set \n {}'.format(X_train.shape[0]))
print('Num rows in testing set \n {}'.format(X_test.shape[0]))

