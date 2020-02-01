import requests
mylink="https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area"
link=requests.get(mylink).text

from bs4 import BeautifulSoup
soup=BeautifulSoup(link,'lxml')
soup

print(soup.prettify())
soup.title
soup.title.string
soup.a
soup.find_all("a")
all_link=soup.find_all("a")
for link in all_link:
    print(link.get("href"))
    
all_tables=soup.find_all("table")
print(all_tables)

right_table=soup.find('table',class_='wikitable sortable')

right_table

table_links=right_table.find_all('a')

table_links

country=[]
for links in table_links:
    country.append(links.get('title'))

country


