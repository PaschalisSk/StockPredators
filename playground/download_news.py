import requests
from data import NYTarticles

NYTarticles = NYTarticles()
NYTarticles.download('microsoft', '2014', '2018') # No article on 31/12/2013
NYTarticles.df.to_csv('../data/nytarticles/microsoft.2013-12-31.2018-12-31.csv',
                      date_format='%Y-%m-%d', index_label='index')
