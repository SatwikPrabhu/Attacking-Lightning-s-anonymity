import csv
import ast
import nested_dict as nd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

file = []

# file = ["results_lnd1.json"]

all_results = []
for i in range(0,5):
    with open(file[i],'r') as json_file:
        results_json = json.load(json_file)
    all_results.append(results_json)

path = []
num_transactions = 0
num_attacked = 0
num_attacks = 0
dest_count = []
dest_count_comp = []
dest_count_incomp = []
source_count = []
source_count_comp = []
source_count_incomp = []
dist_dest = []
pair_found = 0
dist_source = []
num_comp = 0
sing_dest = 0
sing_source = 0
sing_all = 0
sing_any = 0
ads = [2634, 5422, 8075, 5347, 1083, 5093, 4326, 4126, 2836, 5361, 10572, 5389, 3599, 9819, 4828, 3474, 8808, 93, 9530,
       9515, 2163]
ad_attacks = {}
for ad in ads:
    ad_attacks[ad] = 0
for i in all_results:
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
                        for adv in ad:
                            sources = []
                            ad_attacks[int(adv)]+=1
                            ind = k["path"].index(int(adv))
                            dist_dest.append(len(k["path"])-1-ind)
                            dist_source.append(ind)
                            if(k["comp_attack"] == 1):
                                dest_count_comp.append(len(ad[adv]))
                                num_comp+=1
                            else:
                                dest_count_incomp.append(len(ad[adv]))
                            dest_count.append(len(ad[adv]))
                            if(len(ad[adv]) == 1):
                                sing_dest+=1
                            for dest in ad[adv]:
                                for rec in dest:
                                    if dest[rec] == []:
                                        dest[rec] = [k["sender"]]
                                    if int(rec) == k["recipient"] and k["sender"] in dest[rec]:
                                        pair_found+=1
                                    for s in dest[rec]:
                                        sources.append(s)
                            if (k["comp_attack"] == 1):
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



print(num_attacked/num_transactions,num_attacks/num_attacked)
print(np.corrcoef(dest_count,new_dist_dest),np.corrcoef(source_count,new_dist_source))
print(sing_source/num_attacks,sing_dest/num_attacks,sing_any/num_attacks,sing_all/num_attacks)
print(num_comp/num_attacks)
print(pair_found/num_attacks)

#plot1 = sns.ecdfplot(data = new_dest_count_comp,legend='Phase I complete',marker = '|',linewidth = 1.5, linestyle = ':')
plot2 = sns.ecdfplot(data = dest_count_incomp,legend='Phase I incomplete',marker = '|',linewidth = 1.5, linestyle = ':')
#plot1.set(xscale='log')
plot2.set(xscale='log')
# plt.legend(('Phase I complete','Phase I incomplete'),scatterpoints=1,loc='lower right',ncol=1,fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()

# plot1 = sns.ecdfplot(data = new_source_count_comp,legend='Phase I complete',marker = '|',linewidth = 1.5, linestyle = ':')
plot2 = sns.ecdfplot(data = source_count_incomp,marker = '|',linewidth = 1.5, linestyle = ':')
# plot1.set(xscale='log')
plot2.set(xscale='log')
# plt.legend(('Phase I complete','Phase I incomplete'),scatterpoints=1,loc='lower right',ncol=1,fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()


