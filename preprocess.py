'''
Sorts all the images into the appropriate subfolder based on type
'''

#%%
import os
import re
import pandas as pd

#%%
df = pd.read_csv('pokemon_stat.csv')
type_list = df.head(649)['type_1'].to_list()
all_types = list(set(type_list))

#%%
for ptype in all_types:
  os.mkdir('./imgs/%s' % ptype)

#%%
for img in os.listdir('./imgs'):
  if re.match('\d+.png', img):
    index = int(img.split('.')[0])
    os.rename('./imgs/%s' % img, './imgs/%s/%s' % (type_list[index-1], img))