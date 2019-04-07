'''
Sorts all the images into the appropriate subfolder based on type
'''

#%%
import os
import re
import pandas as pd

#%%
df = pd.read_csv('pokemon_stat.csv')
train_list = df.head(649)['type_1'].to_list()
eval_list = df.head(721).tail(721-649)['type_1'].to_list()
all_types = list(set(train_list))

#%%
for ptype in all_types:
  os.mkdir('./imgs/%s' % ptype)

#%%
for ptype in all_types:
  os.mkdir('./eval/%s' % ptype)

#%%
for img in os.listdir('./imgs'):
  if re.match('\d+.png', img):
    index = int(img.split('.')[0])
    os.rename('./imgs/%s' % img, './imgs/%s/%s' % (train_list[index-1], img))

#%%
for img in os.listdir('./eval'):
  if re.match('\d+.png', img):
    index = int(img.split('.')[0])
    os.rename('./eval/%s' % img, './eval/%s/%s' % (eval_list[index-1-649], img))