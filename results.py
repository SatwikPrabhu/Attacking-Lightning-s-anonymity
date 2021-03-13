import csv
import ast
import nested_dict as nd
import matplotlib.pyplot as plt
import numpy as np

file = "transactions_mixed.csv"
file1 = "results_mixed.csv"

transactions = nd.nested_dict()
attacks = nd.nested_dict()
ads = [2634,5422, 8075, 5347, 1083, 5093,4326, 4126, 2836, 5361, 10572,5389, 3599, 9819, 4828, 3474, 8808, 93, 9530, 9515, 2163]
attacked = 0
num_attacks = 0

num_transactions = 0

#store transaction details
with open(file,'r') as csv_file:
    csvreader = csv.reader(csv_file)
    line = 0
    for row in csvreader:
        if(line!=0):
            id = int(row[0])
            transactions[id]["source"] = int(row[1])
            transactions[id]["destination"] = int(row[2])
            transactions[id]["path"] = ast.literal_eval(row[3])
            transactions[id]["delay"] = int(row[4])
            transactions[id]["amount"] = int(row[5])
            transactions[id]["tech"] = int(row[6])
            transactions[id]["success"] = bool(row[7])
            num_transactions+=1
        line+=1


ads1 = dict()
#store details of each attack instance
with open(file1,'r') as csv_file:
    csvreader = csv.reader(csv_file)
    line = 0
    for row in csvreader:
        if(line!=0):
            id = int(row[0])
            ad = int(row[1])
            pot = int(row[2])
            attacks[id][ad][pot] = ast.literal_eval(row[3])
        line+=1

count_dest = []
dest_found = 0
source_found = 0
count_source = []
dists_pot = []
dists_source = []
# generate stats for results of the attack
for id in attacks:
    attacked+= 1
    d = transactions[id]["destination"]
    s = transactions[id]["source"]
    for ad in attacks[id]:
        num_attacks+=1
        pots = 0
        sources = 0
        path = transactions[id]["path"]
        ind = path.index(ad)
        dist_pot = len(path) - 1 - ind
        dist_source = ind
        pot_found = False
        pair_found = False
        for pot in attacks[id][ad]:
            pots+=1
            if pot == d:
                pot_found = True
            sources = 0
            for t in attacks[id][ad][pot]:
                for so in attacks[id][ad][pot][t]:
                    sources+=1
                    if pot == d and so == s:
                        pair_found = True
        dists_pot.append(dist_pot)
        dists_source.append(dist_source)
        if pot_found == True:
            dest_found+=1
        if pair_found == True:
            source_found+=1
        count_dest.append(pots)
        count_source.append(sources)
        print(id,ad,pot_found,pair_found,pots,sources,dist_pot,dist_source)

# number of transactions attacked and the total number of attack instances
print(attacked,num_attacks)

# rate of succeeding to include the correct source and destination in their anonymity sets respectively
print((dest_found/num_attacks),(source_found/num_attacks))

# correlation of hop count from the adversary to the destination and the size of the destination set, and similarly for
# the source
print(np.corrcoef(count_dest,dists_pot),np.corrcoef(count_source,dists_source))

# plot of the size of the source and destination anonymity sets ordered by the transaction id
plt.plot(count_dest,'r--',label= 'target_count')
plt.plot(count_source,'b--',label= 'source_count')
plt.xlabel("Transaction id")
plt.ylabel("Count")
plt.legend(loc = "upper right")
plt.show()


