from random import randint
from bs4 import BeautifulSoup
#from selenium import webdriver
import sqlite3, requests, zlib, json, time, gzip
#from subprocess import Popen, STDOUT
#from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from datetime import timedelta, date, datetime
import pickle
import unicodedata

conn = sqlite3.connect('guardianPrintNewsDump.db')
c = conn.cursor()


start_date = date(2015, 1, 1)
end_date = date(2016, 7, 20)

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def createTable():
     c.execute('''CREATE TABLE IF NOT EXISTS printNewspaper (url TEXT PRIMARY KEY NOT NULL, htmlDump BLOB)''')
     conn.commit()

def convertTofeatures():
	
	'''
	myUrlSet = set()
	with open('myUrlSet.pickle', 'rb') as handle:
		myUrlSet = pickle.load(handle)
	j = 0
	i = 0
	with gzip.open('allGuardianArticlesData_2015to2016.txt.gz','r') as fin:  
		for line in fin:
			try:
				line = json.loads(line)
				Infodata = line['Infodata']['response']['results']
				for r in Infodata:
					publishedwebUrl = r['webUrl']
					if publishedwebUrl in myUrlSet:
						print ("published: " + publishedwebUrl)
						j = j+1
					else:
						print ("not published: " + publishedwebUrl)
						i = i + 1
			except:
				continue
				
	print(j)
	print(i)
	'''
	i = 0
	j = 0
	numberPublished = 0
	notPublished = 0
	myUrlSet = set()
	with open('myUrlSet.pickle', 'rb') as handle:
		myUrlSet = pickle.load(handle)
	
	fpabstract = open('CleanedData/abstract.txt', 'w')
	fpauthor = open('CleanedData/author.txt', 'w')
	fptopics = open('CleanedData/topics.txt', 'w')
	fpProductionOffice = open('CleanedData/productionOffice.txt', 'w')
	fpbody = open('CleanedData/body.txt', 'w')
	fptone = open('CleanedData/toneType.txt', 'w')
	fptagList = open('CleanedData/tagList.txt', 'w')
	fpnumberOfWords = open('CleanedData/numberOfWords.txt', 'w')
	fptrailtext = open('CleanedData/trailtext.txt', 'w')
	fpdate = open('CleanedData/date.txt', 'w')
	fpurl = open('CleanedData/url.txt', 'w')
	fpoutput = open('CleanedData/output.txt', 'w')
	fptimestamp = open('CleanedData/timestamp.txt', 'w')
	
	with gzip.open('allGuardianArticlesData_2015to2016.txt.gz','r') as fin:  
		for line in fin:
			try:
				line = json.loads(line)
				dateToday = line['date']
				Infodata = line['Infodata']['response']['results']
				for r in Infodata:
					flag = 0
					publishedwebUrl = ""
					abstract = ""
					trailtext = ""
					author = ""
					topics = ""
					productionOffice = ""
					body = ""
					toneType = ""
					tagList = ""
					numberOfWords = 0
					timestamp = ""
					flag1 = 0
					flag2 = 0
					flag3 = 0
					for rahead in r['tags']:
						try:
							if rahead['type'] == "contributor" and 'webTitle' in rahead:
								if author == "":
									flag1 = 1
									author = rahead['webTitle']
								else:
									author = author+"&"+rahead['webTitle']
									
							if rahead['type'] == "keyword" and 'webTitle' in rahead:
								if tagList == "":
									flag2 = 1
									tagList = rahead['webTitle']
								else:
									tagList = tagList+"&"+rahead['webTitle']
							
							if rahead['type'] == "tone" and 'webTitle' in rahead:
								if toneType == "":
									flag3 = 1
									toneType = rahead['webTitle']
								else:
									toneType = toneType+"&"+rahead['webTitle']
						except:
							continue
					if flag1 == 0 or flag2 == 0 or flag3 == 0:
						flag = 1
					if 'webPublicationDate' in r:
						timestamp = r['webPublicationDate']
					else:
						flag = 1
					if 'webUrl' in r :
						publishedwebUrl = r['webUrl']
					else:
						flag = 1
					if 'sectionId' in r:
						topics = r['sectionId']
					else:
						flag = 1
					if 'fields' in r :
						if 'standfirst' in r['fields']:
							abstract = r['fields']['standfirst']
						else:
							flag = 1
						if 'productionOffice' in r['fields']:
							productionOffice = r['fields']['productionOffice']
						else:
							flag = 1
						if 'body' in r['fields']:
							body = r['fields']['body']
						else:
							flag = 1
						if 'wordcount' in r['fields']:
							numberOfWords = r['fields']['wordcount']
						else:
							flag = 1
						if 'trailText' in r['fields']:
							trailtext = r['fields']['trailText']
						else:
							flag = 1
					else:
						flag = 1
					
					if flag == 1:
						j = j +1
					else:
						if publishedwebUrl in myUrlSet:
							timestamp = unicodedata.normalize('NFKD', timestamp).encode('ascii','ignore')
							time1 = datetime.strptime(timestamp,'%Y-%m-%dT%H:%M:%SZ')
							time1 = time1.timetuple()
							time1 = time.mktime(time1)
							fptimestamp.write(str(time1)+"\n")
							abstract = unicodedata.normalize('NFKD', abstract).encode('ascii','ignore')
							fpabstract.write(abstract+"\n")
							trailtext = unicodedata.normalize('NFKD', trailtext).encode('ascii','ignore')
							fptrailtext.write(trailtext+"\n")
							author = unicodedata.normalize('NFKD', author).encode('ascii','ignore')
							fpauthor.write(author+"\n")
							topics = unicodedata.normalize('NFKD', topics).encode('ascii','ignore')
							fptopics.write(topics+"\n")
							productionOffice = unicodedata.normalize('NFKD', productionOffice).encode('ascii','ignore')
							fpProductionOffice.write(productionOffice+"\n")
							body = unicodedata.normalize('NFKD', body).encode('ascii','ignore')
							fpbody.write(body+"\n")
							toneType = unicodedata.normalize('NFKD', toneType).encode('ascii','ignore')
							fptone.write(toneType+"\n")
							tagList = unicodedata.normalize('NFKD', tagList).encode('ascii','ignore')
							fptagList.write(tagList+"\n")
							numwor = unicodedata.normalize('NFKD', numberOfWords).encode('ascii','ignore')
							fpnumberOfWords.write(numwor+"\n")
							publishedwebUrl = unicodedata.normalize('NFKD',publishedwebUrl).encode('ascii','ignore')
							fpurl.write(publishedwebUrl+"\n")
							if isinstance(dateToday,unicode):
								dateToday = unicodedata.normalize('NFKD',dateToday).encode('ascii','ignore')
							fpdate.write(dateToday+"\n")  
							fpoutput.write("1\n")
							numberPublished = numberPublished+1
							
						else:
							time1 = datetime.strptime(timestamp,'%Y-%m-%dT%H:%M:%SZ')
							time1 = time1.timetuple()
							time1 = time.mktime(time1)
							fptimestamp.write(str(time1)+"\n")
							abstract = unicodedata.normalize('NFKD', abstract).encode('ascii','ignore')
							fpabstract.write(abstract+"\n")
							trailtext = unicodedata.normalize('NFKD', trailtext).encode('ascii','ignore')
							fptrailtext.write(trailtext+"\n")
							author = unicodedata.normalize('NFKD', author).encode('ascii','ignore')
							fpauthor.write(author+"\n")
							topics = unicodedata.normalize('NFKD', topics).encode('ascii','ignore')
							fptopics.write(topics+"\n")
							productionOffice = unicodedata.normalize('NFKD', productionOffice).encode('ascii','ignore')
							fpProductionOffice.write(productionOffice+"\n")
							body = unicodedata.normalize('NFKD', body).encode('ascii','ignore')
							fpbody.write(body+"\n")
							toneType = unicodedata.normalize('NFKD', toneType).encode('ascii','ignore')
							fptone.write(toneType+"\n")
							tagList = unicodedata.normalize('NFKD', tagList).encode('ascii','ignore')
							fptagList.write(tagList+"\n")
							numwor = unicodedata.normalize('NFKD', numberOfWords).encode('ascii','ignore')
							fpnumberOfWords.write(numwor+"\n")
							publishedwebUrl = unicodedata.normalize('NFKD',publishedwebUrl).encode('ascii','ignore')
							fpurl.write(publishedwebUrl+"\n")
							if isinstance(dateToday,unicode):
								dateToday = unicodedata.normalize('NFKD',dateToday).encode('ascii','ignore')		
							fpdate.write(dateToday+"\n")
							fpoutput.write("0\n")
							notPublished = notPublished+1
						i = i+1
				
			except:
				continue
	
	fpabstract.close()
	fpauthor.close()
	fptopics.close()
	fpProductionOffice.close()
	fpbody.close()
	fptone.close()
	fptagList.close()
	fpnumberOfWords.close()
	fptrailtext.close()
	fpoutput.close()
	fpdate.close()
	fpurl.close()
	fptimestamp.close()
	
	print("not proper: "+str(j))
	print("proper: "+ str(i))
	print("number Published: "+ str(numberPublished))
	print("not Published: "+ str(notPublished))
		

def main():
	'''
    xvfbCmdLine = "/usr/bin/Xvfb :11 -nolisten tcp -ac -cc 4 -screen 0 1200x800x24"
    xvfbProcess = Popen(xvfbCmdLine, shell=True, stderr=STDOUT)
     
    binary = FirefoxBinary(firefox_path="../Personalized-Reco/firefox/firefox")
    binary.add_command_line_options('--display=:11.0')
    driver = webdriver.Firefox(firefox_binary=binary)
    driver.set_window_size(1200, 800) # optional
     
    for single_date in daterange(start_date, end_date):
        try:
            if single_date.month == 1:
                url = "http://www.theguardian.com/theguardian/{0}/jan/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 2:
                 url = "http://www.theguardian.com/theguardian/{0}/feb/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 3:
                 url = "http://www.theguardian.com/theguardian/{0}/mar/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 4:
                 url = "http://www.theguardian.com/theguardian/{0}/apr/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 5:
                 url = "http://www.theguardian.com/theguardian/{0}/may/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 6:
                 url = "http://www.theguardian.com/theguardian/{0}/jun/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 7:
                 url = "http://www.theguardian.com/theguardian/{0}/jul/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 8:
                 url = "http://www.theguardian.com/theguardian/{0}/aug/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 9:
                 url = "http://www.theguardian.com/theguardian/{0}/sep/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 10:
                 url = "http://www.theguardian.com/theguardian/{0}/oct/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 11:
                 url = "http://www.theguardian.com/theguardian/{0}/nov/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))
            elif single_date.month == 12:
                 url = "http://www.theguardian.com/theguardian/{0}/dec/{1}".format(single_date.strftime('%Y'), single_date.strftime('%d'))

            cursor = c.execute('SELECT COUNT(*) FROM printNewspaper WHERE url = (?)', (url,))
            articlecount = 0
            for row in cursor:
                articleCount = int(row[0])

            if articleCount == 0:
                driver.get(url)
                content = driver.page_source
                compressedHTMLDump = buffer(zlib.compress(content.encode('utf8')))
                c.execute('INSERT INTO printNewspaper VALUES (?, ?)', (url, compressedHTMLDump))
                time.sleep(randint(120, 300))

        except Exception as e:
               print e

    conn.close()
    driver.quit()
    xvfbProcess.kill()
    xvfbProcess.wait()
    '''
	
if __name__ == "__main__":
	#createTable()
	#
	#timestamp = "2015-01-01T18:30:00Z"
	#time1 = datetime.now()
	#time1 = time1.timetuple()
	#print(time.mktime(time1))
	#main()
	convertTofeatures()