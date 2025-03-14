import pandas as pd
import os
import re

#!python /kaggle/input/wikiextractor-files/WikiExtractor/WikiExtractor.py /kaggle/input/rowiki-latest-unzipped/rowiki-20240201-pages-articles.xml -o ./pages_final

lista = []

root = "/kaggle/input/extracted-articles-wiki/wiki_dump_extracted"
folders = os.listdir(root)
folders.sort()


for folder in folders:                                               
    crt_folder = os.path.join(root, folder)
    
    files = os.listdir(crt_folder)
    files.sort()

    for file in files:
        path = os.path.join(crt_folder, file)

        with open(path,'r') as f:
            articles = re.findall(r'(<doc[\S\s]*?</doc>)', f.read())

        for i in range(0, len(articles)):

            x = re.search(r'=====', articles[i])

            if x==None:
                continue

            summary = articles[i][:x.start()]
            summary = re.sub(r"<.*?>\s", '', summary)

            content = articles[i][x.start()+1:-6]

            idx = re.search(r'[\n]', summary).start()
            title = summary[:idx]
            summary = summary[idx+1:]
            
            if re.match("^[0-9 ]+$", title):    #Articolele ce contin doar numere in titlu nu sunt bune
                continue
            if title.split()[0]=="Listă":       #Articolele cu "Lista" in nume nu sunt bune
                continue

            summary = re.sub(r"&lt.*?&gt;", "", summary)
            summary = re.sub(r"&lt;", "", summary)
            summary = re.sub(r"&gt;", "", summary)
            summary = re.sub(r"[\n]", "", summary)
            
            summary = re.sub(r"\(.*?\)", "", summary)
            summary = re.sub(r"\[\[.*?\]\]", "", summary)
            summary = re.sub(r"\"*", "", summary)

            content = re.sub(r"&lt.*?&gt;", "", content)
            content = re.sub(r"=+.*[\n]", "", content)
            content = re.sub(r"&lt;", "", content)
            content = re.sub(r"&gt;", "", content)
            content = re.sub(r"[\n]", "", content)
            
            content = re.sub(r"\(.*?\)", "", content)
            content = re.sub(r"\[\[.*?\]\]", "", content)
            content = re.sub(r"\"*", "", content)

            if len(summary) < 400:       ##Elimina articole cu mai putin de x caractere in rezumat
                continue
            if len(content) < 1000:       ##Elimina articole cu mai putin de y caractere in continut
                continue

            lista.append([title, summary, content]) 


dataset = pd.DataFrame(lista, columns=['Title', 'Summary', 'Content'])

dataset.head()

dataset.to_csv("set.csv")

#print(dataset.iloc[2])
#dataset.head(60)
