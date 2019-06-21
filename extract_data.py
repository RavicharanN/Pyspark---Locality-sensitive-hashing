import pandas
import pickle
import csv 
import numpy as np
from datasketch import MinHash, MinHashLSH

csv_reader = pandas.read_csv('data/string_study.csv', delimiter=',', header=None)

final_arr = csv_reader.values

final_arr = np.array(final_arr)

# print(final_arr)

raw_msgs = final_arr[1:,-1]
print(raw_msgs.shape)
message_set = []

for item in raw_msgs:
    # print(item)
    set_of_split_arr = set(item.split(' '))
    # print(set_of_split_arr)
    message_set.append(set_of_split_arr)

print(message_set[1])

m = []
for i in range (0, len(message_set)):
    m_temp = MinHash(num_perm=128)
    m.append(m_temp)

i = 0
for message in message_set:
    for word in message:
        m[i].update(word.encode('utf-8'))
    i = i + 1

# Create an LSH index
lsh = MinHashLSH(threshold=.4, num_perm=128)
for i in range (1, len(m)):
    string = "m" + str(i)
    lsh.insert(string, m[i])
result = lsh.query(m[0])
print("Items with Jacard similarity > .4", result)