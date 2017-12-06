import urllib2
import csv,os,json, re
import requests
import random
from dateutil import parser as dateparser
from bs4 import BeautifulSoup

def get_headers():
    # choose a user agent to use for this request
    user_agents = open('Data/user_agents.txt').read().splitlines()
    useragent = random.choice(user_agents)
    header = {'User-agent':useragent}
    return header

def download(url, type):
	page = requests.get(url, headers = get_headers())
	soup = BeautifulSoup(page.content,'lxml')
	hrefs = soup.find_all('a')
	for a in hrefs:
		name = a.text
		req = urllib2.Request(a['href'], None, get_headers())
		response = urllib2.urlopen(req)
		file = open('Data/Mnih/' + type + name,'w')
		file.write(response.read())
		file.close()
		print( "Downloaded: "+ name)


url = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html'
download(url, type = 'Training/Input/')
print('Training/Input DONE.')

url = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html'
download(url, type='Training/Target/')
print('Training/Target/ DONE.')

url ='https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html'
download(url, type='Validation/Input/')
print('Validation/Input/ DONE.')

url = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html'
download(url, type='Validation/Target/')
print('Validation/Target/ DONE.')

url = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html'
download(url, type='Test/Input/')
print('Test/Input/ DONE.')

url = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html'
download(url, type='Test/Target/')
print('Test/Target/ DONE.')
