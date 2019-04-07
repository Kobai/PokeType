#%%
import urllib.request
import os

#%%
os.mkdir('./imgs')
os.mkdir('./eval')
os.mkdir('./predict')
os.mkdir('./predict/imgs')

#%%
for i in range(649):
  num = '%03d' % (i+1)
  urllib.request.urlretrieve('https://assets.pokemon.com/assets/cms2/img/pokedex/full/%s.png' % num, './imgs/%s.png' % str(i+1))

#%%
for i in range(650,722):
  num = '%03d' % i
  urllib.request.urlretrieve('https://assets.pokemon.com/assets/cms2/img/pokedex/full/%s.png' % num, './eval/%s.png' % str(i))