
#PLOT THE DATA 
datetimeindex = dataset.indexes['time'].to_datetimeindex()
plt.plot(datetimeindex,dataset['pr'].to_numpy(),color='blue',label='adj')
plt.legend()
plt.show()