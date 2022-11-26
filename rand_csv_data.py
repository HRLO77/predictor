import csv
import random

arr = []
cur_val = 1000
until_k = False
m = 0
for i in range((365*20)):
    if ((i//365) % 2) == 0:
        m = 10
        if until_k:
            if cur_val >= 1500:
                until_k = False
                continue
            cur_val += random.randint(1,5)
            arr += [{cur_val}]
        else:
            if cur_val <= 0:
                until_k = True
                continue
            cur_val -= random.randint(1,5)
            if cur_val < 0:
                cur_val = 0
            arr += [{cur_val}]
    else:
        
        if until_k:
            if cur_val >= 1500:
                until_k = False
                continue
            cur_val += m
            m += random.randint(1,5)
            # arr += [{cur_val}]
        if not until_k:
            # m=10
            if cur_val <= 0:
                until_k = True
                continue
            cur_val -= m
            m += random.randint(1,5)
            if cur_val < 0:
                m = random.randint(0, m)
                cur_val = 0
            # arr += [{cur_val}]
# arr = numpy.array(arr)
writer = csv.writer(open('./data.csv', 'w+'))
writer.writerow('X')
writer.writerows(arr)