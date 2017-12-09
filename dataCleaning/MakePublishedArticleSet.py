from random import randint
from bs4 import BeautifulSoup
#from selenium import webdriver
import sqlite3, requests, zlib, json, time, gzip
#from subprocess import Popen, STDOUT
#from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from datetime import timedelta, date
import pickle
import unicodedata


conn = sqlite3.connect('guardianPrintNewsDump.db')
c = conn.cursor()
	
def main():	
	
	cursor = c.execute('SELECT * FROM printNewspaper')
	
	myUrlDateArticle = dict()
	l1 = 0
	k1 = 0
	for row in cursor:
		#print row
		content = zlib.decompress(row[1])
		#print content
		soup = BeautifulSoup(content, 'html.parser')
		
		
		print(row[0])
		printedUrlFullData = soup.find_all("a", { "data-link-name" : "article" ,"tabindex" : "-1"});
		for link in printedUrlFullData:
			l1 = l1 +1
			printedwebUrl = link.get('href')
			if printedwebUrl in myUrlDateArticle:
				k1 = k1+1
				print(l1)
				myUrlDateArticle[printedwebUrl].append(row[0])
			else:
				#print(k1)
				#print("here")
				myUrlDateArticle[printedwebUrl] = [row[0]]
	with open('myUrlSet.pickle', 'wb') as handle:
		pickle.dump(myUrlDateArticle,handle)
	
	
if __name__ == '__main__':
    main()