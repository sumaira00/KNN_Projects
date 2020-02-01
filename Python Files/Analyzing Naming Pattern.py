import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

names=[]
for year in range(1880,2019):
    names.append(pd.read_csv(f'C:search-ms:displayname=Search%20Results%20in%20Downloads&crumb=location:C%3A%5CUsers%5Csaim-zain%5CDownloads\us-baby-names.zip\NationalNames.csv',names=['Names','Gender','Frequncy']))
    names[-1]['Year']=year
    
    
names_total=pd.concat(names)
names_total.columns
names_total.head()

names_by_year=names_total.groupby(by='Gender').Frequncy.sum()
names_by_year
names_by_year.plot(kind='bar',stacked=True,color='blue');

names_by_year.plot(kind='box',stacked=True,color='green');

total_by_gender=names_total.pivot_table('Frequncy',index='Year',columns='Gender',aggfunc=sum)

total_by_gender.plot(title='Total Births by sex and year')

Males=names_total[names_total['Gender']=='M']
Females=names_total[names_total['Gender']=='F']

total_by_gender.plot.bar(stacked=True,rot=0)

total_by_gender.plot.box(stacked=True)

Total_Names=names_total.pivot_table('Frequncy',index='Year',columns='Names',aggfunc=sum)

Total_Names.head()

allyears_indexed = names_total.set_index(['Gender','Names','Year']).sort_index(level=0)

allyears_indexed.head()

M2018=allyears_indexed.loc['M',:,2018].sort_values('Frequncy',ascending=False).head()
F2018=allyears_indexed.loc['F',:,2018].sort_values('Frequncy',ascending=False).head()

M2018.reset_index().drop(['Gender','Frequncy','Year'],axis=1).head(3)

F2018.reset_index().drop(['Gender','Frequncy','Year'],axis=1).head(3)
F2018.columns=['Year']

F2018
def sort_yearly(sex,year):
    simple = allyears_indexed.loc[sex,:,year].sort_values('Frequncy',ascending=False).reset_index()
    simple = simple.drop(['Gender','Year','Frequncy'],axis=1).head(10)
    simple.columns = [year]
    simple.index = simple.index + 1
    return simple

sort_yearly('M',2018)

def sort_definite(sex,year0,year1):
    years = [sort_yearly(sex,year) for year in range(year0,year1+1)]
    return years[0].join(years[1:])

sort_definite('M',2016,2018).stack().value_counts()

