import pandas as pd

df = pd.read_table('SMSSpamCollection',
		sep='\t',
		header=None,
		names=['label', 'sms_message'])
print('head:'+str(df.head()))

df['label'] = df.label.map({'ham':0, 'spam':1})
print('shape: '+str(df.shape))
print('head2: '+str(df.head()))
