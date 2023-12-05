#link dataset: https://data.world/data-society/air-traffic-passenger-data
#nume fisier dataset: Air_Traffic_Passenger_Statistics.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections

plt.style.use('fivethirtyeight')
import plotly.offline as py
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from datetime import datetime


# importare si afisare set de date din fisierul csv
df = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')
print(f"Forma setului de date: {df.shape}")
print(df.head(5))


# aflare mai multe detalii despre setul de date(data de inceput, data finala, durata in zile samd)
time_begin = df.loc[:,"Activity Period"].min() # data de inceput
time_end = df.loc[:,"Activity Period"].max() # data de sfarsit din dataset(maxima)

date_begin = datetime.strptime(str(time_begin), '%Y%m')
date_end = datetime.strptime(str(time_end), '%Y%m')
time_range = date_end - date_begin # intervalul de timp dintre maxim si minim(max-min)

print("Prima data(minima): ", str(time_begin)[:4], "/", str(time_begin)[4:])
print("Ultima data(maxima): ", str(time_end)[:4], "/", str(time_end)[4:])
print("Intervalul de timp in zile: ", time_range.days)

# heatmap pentru toate companiile in functie de numarul de zboruri, regiunea geografica si numarul de pasageri, per an
PAX_airline_yr = df.groupby(["Year","Operating Airline"])["Passenger Count"].sum().divide(1000)
PAX_airline_yr = PAX_airline_yr.reset_index()
pivot_1 = PAX_airline_yr.pivot_table(values="Passenger Count",index="Operating Airline",columns="Year", fill_value=0)
pivot_1.loc["United Airlines",:] = pivot_1.loc["United Airlines",:] + pivot_1.loc["United Airlines - Pre 07/01/2013",:]
pivot_1.drop("United Airlines - Pre 07/01/2013",axis=0, inplace=True)

# eliminarea companiilor mici(cu zboruri rare/putine)
dropped = pivot_1[pivot_1.sum(axis=1)<13]
pivot_1 = pivot_1.drop(dropped.index,axis=0)

sns.set(font_scale=0.7)
fig1 = plt.figure(figsize=(12,20))
p1 = sns.heatmap(pivot_1, annot=True, linewidths=.5, vmin=100, vmax=1000, fmt='.0f', cmap=sns.cm.rocket_r)
p1.set_title("Numar de pasageri transportati(de ordinul miilor)", fontweight="bold")
p1.set_yticklabels(p1.get_yticklabels(), rotation=0)
plt.tight_layout()


# aflare top 5 companii aeriene in functie de media pasagerilor transportati/an si reprezentarea acestora ca procent
avg_airline = pivot_1.mean(axis=1)
TOP5_airlines = avg_airline.nlargest(5).to_frame().mul(1000).astype("int64")
TOP5_airlines.columns = ["Media pasagerilor/An"]
sum_of_all = TOP5_airlines.loc[:,"Media pasagerilor/An"].sum()
TOP5_airlines.loc[:,"Procent [in pct]"] = TOP5_airlines.loc[:,"Media pasagerilor/An"].div(sum_of_all).mul(100).round(1)
TOP5_airlines


# aflare pondere a regiunilor geografice raportate la traficul total
grph = df.groupby(["Year","GEO Region"])["Passenger Count"].sum()
grph = grph.reset_index()

pivot_5 = grph.pivot_table(values="Passenger Count",index="Year",columns="GEO Region", fill_value=0)
pivot_5 = pivot_5.drop("US",axis=1)
pivot_5["Total"] = pivot_5.sum(axis=1)

for col in pivot_5.columns[:-1]:
        pivot_5["Share " + str(col)]=pivot_5[col]/pivot_5["Total"]

ratios_5 = pivot_5.iloc[:,-8:]
ratios_5.plot(figsize=(12,8), title="Ponderea regiunilor geografice raportate la traficul total")

CotaTrafic = df.groupby(["Activity Period","GEO Summary"])["Passenger Count"].sum()
CotaTrafic = CotaTrafic.reset_index()

pivot_4 = CotaTrafic.pivot_table(values="Passenger Count",index="Activity Period",columns="GEO Summary", fill_value=0)
pivot_4["Share"] = (pivot_4["International"]/(pivot_4["Domestic"]+pivot_4["International"])).mul(100)
mean_CotaTrafic = pivot_4["Share"].mean()

fig4, ax4 = plt.subplots(figsize=(12,6))
plt.title("Cota de trafic al pasagerilor în zboruri internationale vs. domestice(în%)", fontdict={'fontsize':13,'fontweight' : "bold"})
plt.ylim(0,100)

pivot_4["Share"].plot(ax=ax4, c="white")
ax4.fill_between(pivot_4.index, 100, color='#f48342')
ax4.fill_between(pivot_4.index, pivot_4["Share"], color='#4189f4')
ax4.axhline(mean_CotaTrafic, c="black", linestyle="--")
plt.show()

# countplot in functie de fiecare companie si zborurile efectuate de acestea
plt.figure(figsize = (15,10))
sns.countplot(x = "Operating Airline", palette = "Set3",data = df)
plt.xticks(rotation = 90)
plt.ylabel("Numar de zboruri efectuate")
plt.show()

# countplot in functie de regiunea geografica si zborurile efectuate in aceste regiuni
plt.figure(figsize = (15,10))
sns.countplot(x = "GEO Region", palette = "Set3",data = df)
plt.xticks(rotation = 90)
plt.ylabel("Numar de zboruri efectuate")
plt.show()

airline_count = df["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)
passenger_count = df.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)


df_1 = airline_count + passenger_count
df_1.sort_values(ascending = False, inplace = True)
outliers = df_1.head(2).index.values
airline_count = airline_count.drop(outliers)
airline_count.sort_index(inplace=True)
passenger_count = passenger_count.drop(outliers)
passenger_count.sort_index(inplace = True)
x = airline_count.values
y = passenger_count.values


# utilizare "elbow method" pentru aflarea numarului optim de clustere
X = np.array(list(zip(x,y)))
inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
plt.plot(range(2,10), inertias, "o-g")
plt.xlabel("Numar de clustere")
plt.ylabel("Inertia")
plt.title("Metoda 'cotului' pentru numarul optim de clustere")
plt.show()


# aplicarea algoritmului KMeans cu numarul optim de clustere si aflarea etichetelor fiecarei companii/inregistrari
kmeans = KMeans(n_clusters=6)
y_kmeans = kmeans.fit_predict(X)


# plotare a clusterelor si centroizilor in functie de companie/numar de pasageri
plt.figure(figsize = (15,15))
plt.xlabel("Zboruri efectuate")
plt.ylabel("Pasageri")


# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=300, cmap='Set1') #, cmap='Set1'
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 300, c = 'green', label = 'Companie medie(P medii, Z putine)')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 300, c = 'yellow', label = 'Companie mare(P multi, Z medii)')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 300, c = 'cyan', label = 'Companie mica(PZ f mici)')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 300, c = 'magenta', label = 'Companie mare(PZ medii)')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 300, c = 'orange', label = 'Companie mare(PZ foarte mari)')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 300, c = 'blue', label = 'Companie mica(P mici-medii ,Z putine)')

for i, txt in enumerate(airline_count.index.values):
    plt.annotate(txt, (X[i,0], X[i,1]), size = 7)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black' , label = 'centeroid')
plt.legend()
plt.show()

# afisare componente pentru fiecare cluster
from collections import Counter
y_kmeans2 = list(set(y_kmeans))
print(y_kmeans2)
# print(f"Componente cluster 0:\n\tX[0](k$):{X[y_kmeans == 0, 0]}\n\tX[1]:{X[y_kmeans == 0, 1]}")
for i in y_kmeans2:
    print(f"Componente cluster {i}:\n\tX[0]-airline_counts:{X[y_kmeans == i, 0]}\n\tX[1]-passenger_count:{X[y_kmeans == i, 1]}")
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black' , label = 'centeroid')

print("---" * 5 + "Dendrograma pentru clusterele obtinute " + "---" * 5)
fig = ff.create_dendrogram(X, labels = y_kmeans)
fig.update_layout(width=800, height=500)
fig.show()


# afisare piechart-uri cu procentele aferente fiecarei companii(din top 10) si fiecarui tip de zbor
df.loc[df['Operating Airline'] == 'United Airlines - Pre 07/01/2013', 'Operating Airline'] = 'United Airlines'

operating_airlines = collections.Counter(df['Operating Airline'])
geo_summary = collections.Counter(df['GEO Summary'])
# print(operating_airlines.most_common(5))
# print(geo_summary.most_common())

print(f"Top 10 companii comune: {operating_airlines.most_common(10)}")
print(f"Ultimele 10 companii comune: {operating_airlines.most_common()[-10:]}")

oa = operating_airlines.most_common(10)
oa_least_common = operating_airlines.most_common()[-10:]
gs = geo_summary.most_common()


operating_airlines_cnt = [elem[1] for elem in oa]
operating_airlines_names = [elem[0] for elem in oa]
geo_summary_cnt = [elem[1] for elem in gs]
geo_summary_names = [elem[0] for elem in gs]

operating_airlines_least_common_cnt = [elem[1] for elem in oa_least_common]
operating_airlines_least_common_names = [elem[0] for elem in oa_least_common]


fig, (ax1) = plt.subplots(1,1)
ax1.pie(operating_airlines_cnt, labels=operating_airlines_names, autopct='%1.1f%%')
ax1.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
ax1.set_title("Piechart pentru top 10 companii comune")

fig, (ax2) = plt.subplots(1, 1)
ax2.pie(geo_summary_cnt, labels=geo_summary_names, autopct='%1.1f%%')
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
ax2.set_title("Piechart pentru tipurile de zbor")


