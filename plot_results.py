import matplotlib.pyplot as plt
import csv
plt.style.use('seaborn-whitegrid')
import math
year = []
gpp = []
agb = []
with open('output.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        x = row[0].split(',')
        if(len(x) > 2):
                year.append(float(x[0]))
                gpp.append(float(x[1]))
                agb.append(float(x[2]) * 1000 / math.pow(10,12))

cesm_agb = []
with open('output/output_cesm_only.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        x = row[0].split(',')
        cesm_agb.append(float(x[1]) * 1000 / math.pow(10,12))
cesm_gpp = []
with open('output_cesm_gpp_only.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        x = row[0].split(',')
        cesm_gpp.append(float(x[1]))
# plt.plot(year,gpp, label='gpp')
plt.plot(year,gpp, label='constrained gpp')
plt.plot(year,cesm_gpp, label='cesm gpp')
plt.xlabel('year')
plt.ylabel('kgC/100km/year ')
plt.legend()
plt.savefig('output_gpp.png')