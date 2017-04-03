
# coding: utf-8

# In[41]:

from __future__ import division
from __future__ import print_function
import math


# In[42]:

def distance(x1, x2, y1, y2):
    d = math.fabs(x1-x2) + math.fabs(y1-y2)
    return d


# In[72]:

data = {1: [2, 3, 1],
        2: [4, 4, 1],
        3: [4, 5, 0],
        4: [6, 3, 1],
        5: [8, 3, 0],
        6: [8, 4, 0]}
ks = [1, 2, 3]
accuracy = 0.0
pref_k = 0
for k in ks:
    correct = 0
    for inst in data.keys():
        # testing instance
        test = data[inst]
        temp = dict(data)
        del temp[inst]
        # get dist from testing instance to all training instances
        dist = {}
        for key, value in temp.iteritems():
            x1 = test[0]
            y1 = test[1]
            x2 = value[0]
            y2 = value[1]
            d = distance(x1, x2, y1, y2)
            dist[key] = d
        # get k nearest neighbor instances
        nn = []
        for i in range(1, k+1):
            near_val = min(dist.values())
            near_inst = [ins for ins, d in dist.iteritems() if d == near_val]
            for num in near_inst:
                # tie breaking is just first-in wins (i.e. lower instance #)
                if len(nn) == k:
                    break
                else:
                    pn = temp[num]
                    nn.append(pn[2])
                    dist.pop(num)
        # counter for if nearest neighbor(s) produces correct prediction
        for n in nn:
            if test[2] == n:
                correct = correct + 1
    frac = correct / (k*6)
    print(k, correct, frac)


# In[ ]:



