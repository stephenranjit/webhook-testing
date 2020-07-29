#!/usr/local/bin/python3


#import requests

from pyquery import PyQuery as pq
#from bs4 import BeautifulSoup
import firebase
import requests
import pprint
import json
import sys
import re

#def createMapping():


def getKeyTherapeuticBenefits(herb):
    htmlContent = requests.get(url+herb, stream=True)
    #htmlContent = htmlContent.text()
    i = j = 0
    endOfList = 0
    htmlContentArray = list(htmlContent.iter_lines())
    benefits = []
    for line in htmlContentArray:
        if line:
            htmlString = line.decode('utf-8')
            if re.search('Key therapeutic benefits:', htmlString):
                #print('found at line # '+ str(i))
                j = i
                while not endOfList:
                    #print("line # "+str(j))
                    #print("eol:"+str(endOfList))
                    #print("scanning line :"+htmlContentArray[j].decode('utf-8'))
                    if re.search('<ul type="disc">', htmlContentArray[j].decode('utf-8')):
                        k = j
                        while not endOfList:
                            m = re.search('<li>([^<>]+)<', htmlContentArray[k].decode('utf-8'))
                            if m:
                                #print(m.group(1))
                                benefits.append(m.group(1))
                            elif re.search('</ul>', htmlContentArray[k].decode('utf-8')):
                               endOfList= 1
                            k += 1
                    j += 1

        i += 1

    #print(benefits)
    return benefits
    #sys.exit(0)

    

#Key therapeutic benefits:


url = 'https://herbfinder.himalayawellness.in/'
#r = requests.get(url)
#print(r.text)
herbPages = []
herbBenefitsDict = {}

e = pq(url=url+'latin.htm')
#benefits = e('<h4><strong>Key therapeutic benefits:</strong></h4>')
#pprint.pprint(e.after('h4'))
for link in e('li.all a'):
    #print(link.attrib['href'])
    herbPages.append(link.attrib['href'])


#pprint.pprint(herbPages)

for herb in herbPages:
    #print(herb)
    herbBenefitsDict[herb.replace('.htm', '')] = getKeyTherapeuticBenefits(herb)
    #break
    #f = firebase.FirebaseDB()
    #f.insert(herbBenefitsDict)
    #sys.exit(0)
#pprint.pprint(herbBenefitsDict.values())

for val in herbBenefitsDict.values():
    for ele in val:
        print(ele)


