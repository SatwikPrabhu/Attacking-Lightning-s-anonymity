import csv
import ast
import nested_dict as nd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import networkx as nx
import populate_graph as pg

file = "results.json"

# Open the results file and transfer the in an array
results = []
with open(file,'r') as json_file:
    results_json = json.load(json_file)
results.append(results_json)


path = []

# Number of Transactions
num_transactions = 0

# Number of transactions attacked
num_attacked = 0

# Total number of attack instances
num_attacks = 0

# Array storing the number of recipients for each attack instance, followed by those that that had phase I completed and not respectively
dest_count = []
dest_count_comp = []
dest_count_incomp = []

# Arrays storing the number of senders for each attack instance, followed by those that that had phase I completed and not respectively
source_count = []
source_count_comp = []
source_count_incomp = []

# Arrays storing the distances of the recipient and the sender from the adversary respectively
dist_dest = []
dist_source = []

# Number of attack instances in which the sender and recipient pair was successfully found
pair_found = 0

# Number of attack instances that completed phase I
num_comp = 0

# Number of attack instances for which the size of the recipient set was 1 and similarly for the sender
sing_dest = 0
sing_source = 0

# Number of attack instances having both the sender and recipient sets singular
sing_all = 0

# Number of attack instances having atleast one of the sender and recipient sets singular
sing_any = 0
ads = [2634, 5422, 8075, 5347, 1083, 5093, 4326, 4126, 2836, 5361, 10572, 5389, 3599, 9819, 4828, 3474, 8808, 93, 9530,
       9515, 2163]

# Dictionary for storing the number of attack instances of each adversary
ad_attacks = {}
for ad in ads:
    ad_attacks[ad] = 0
    
# Go over the results and update each of the above variables for each attack instance
for i in results:
    for j in i:
        for k in j:
            if k["path"]!=path:
                path = k["path"]
                num_transactions+=1
                if k["attacked"]>0:
                    num_attacked+=1
                    anon_sets = k["anon_sets"]
                    for ad in anon_sets:
                        num_attacks+=1
                        num = -1
                        for adv in ad:
                            sources = []
                            ad_attacks[int(adv)]+=1
                            num+=1
                            for dest in ad[adv]:
                                for rec in dest:
                                    for tech in dest[rec]:
                                        if int(rec) == k["recipient"] and k["sender"] in dest[rec][tech]:
                                            pair_found+=1
                                        for s in dest[rec][tech]:
                                            sources.append(s)
                            if len(set(sources))>0:
                                ind = k["path"].index(int(adv))
                                dist_dest.append(len(k["path"])-1-ind)
                                dist_source.append(ind)
                                if(k["comp_attack"][num] == 1):
                                    dest_count_comp.append(len(ad[adv]))
                                    num_comp+=1
                                else:
                                    dest_count_incomp.append(len(ad[adv]))
                                dest_count.append(len(ad[adv]))
                                if(len(ad[adv]) == 1):
                                    sing_dest+=1
                                if (k["comp_attack"][num] == 1):
                                    source_count_comp.append(len(set(sources)))
                                else:
                                    source_count_incomp.append(len(set(sources)))
                                source_count.append(len(set(sources)))
                                if(len(set(sources))==1):
                                    sing_source+=1
                                if(len(ad[adv]) ==1) or(len(set(sources))==1):
                                    sing_any+=1
                                if (len(ad[adv]) == 1) and (len(set(sources)) == 1):
                                    sing_all += 1
# print(num_transactions,num_attacked,num_attacks,pair_found)
# print(source_count)
# print(dest_count)

# Print the metrics
print(num_attacked/num_transactions,num_attacks/num_attacked)
print(np.corrcoef(dest_count,dist_dest),np.corrcoef(source_count,dist_source))
print(sing_source/num_attacks,sing_dest/num_attacks,sing_any/num_attacks,sing_all/num_attacks)
print(num_comp/num_attacks)
print(num_transactions)


# Plot the sender and recipient anonymity sets respectively
plot1 = sns.ecdfplot(data = dest_count_comp,legend='Phase I complete',marker = '|',linewidth = 1.5, linestyle = ':')
plot2 = sns.ecdfplot(data = dest_count_incomp,legend='Phase I incomplete',marker = '|',linewidth = 1.5, linestyle = ':')
plot1.set(xscale='log')
plot2.set(xscale='log')
plt.legend(('Phase I complete','Phase I incomplete'),scatterpoints=1,loc='lower right',ncol=1,fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()

plot1 = sns.ecdfplot(data = source_count_comp,legend='Phase I complete',marker = '|',linewidth = 1.5, linestyle = ':')
plot2 = sns.ecdfplot(data = source_count_incomp,legend='Phase I incomplete',marker = '|',linewidth = 1.5, linestyle = ':')
plot1.set(xscale='log')
plot2.set(xscale='log')
plt.legend(('Phase I complete','Phase I incomplete'),scatterpoints=1,loc='lower right',ncol=1,fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()


