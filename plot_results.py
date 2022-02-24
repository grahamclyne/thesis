import matplotlib.pyplot as plt
import csv
plt.style.use('seaborn-whitegrid')

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
                agb.append(float(x[2]))

cesm_agb = []
with open('output_cesm_only.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        x = row[0].split(',')
        cesm_agb.append(float(x[1]))


# plt.plot(year,gpp)
plt.plot(year,agb)
plt.plot(year,cesm_agb)

plt.show()