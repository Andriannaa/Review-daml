import pandas as pd
import numpy as np
import random
import pdb


csv_data = pd.read_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/Video_Games_filter.csv')
file = pd.DataFrame(csv_data)
rating = file[['user','item','rating','time']]
userId = rating['user']
itemId = rating['item']


# split_data
user = userId.unique()
rating_dict= dict()
for j in range(len(userId)):
    u = userId[j]
    i = itemId[j]
    rating_dict.setdefault(u,[]).append(i)

traindata = dict()
testdata = dict()
valdata = dict()
# 遍历rating_dict的每一个key
#train：test=8:2，因为过滤了交互小于5的user，所以test的user对应的len(item)不会等于0
for i in rating_dict.keys():
    traindata[i] = random.sample(rating_dict[i],int(len(rating_dict[i])*0.8))
    testdata[i] = set(rating_dict[i]).symmetric_difference(traindata[i])  # 转化为set格式，因为list have not attribute symmetric_difference 
# val为train的0.1

for i in traindata.keys():
    # 如果val的item长度小于1,则去掉这个user
    if len(traindata[i])<10:
        continue
    else:
        valdata[i] = random.sample(traindata[i],int(len(traindata[i])*0.1))
    #或者： valdata[i] = random.sample(traindata[i],max(1,int(len(traindata[i])*0.1)))保证val的每个user有item  


# save rating to csv
test_csv = pd.DataFrame(columns=['user','item','rating','time'])
for i in testdata.keys():
    test_u = rating[rating['user'].isin([i])]
    item = testdata[i]
    test_ui = test_u[test_u['item'].isin(list(item))]
    test_csv =  pd.concat([test_csv,test_ui],axis=0,ignore_index=False)  #dataframe上下合并,ignore_index=False保留file原来的index号
print("test_csv finished!")
train_csv = rating.drop(test_csv.index)  # rating去除test_csv的index对应的所有行
print("train_scv finished!")

val_csv = pd.DataFrame(columns=['user','item','rating','time'])
for i in valdata.keys():
    val_u = train_csv[train_csv['user'].isin([i])]
    item = valdata[i]
    val_ui = val_u[val_u['item'].isin(list(item))]
    val_csv =  pd.concat([val_csv,val_ui],axis=0,ignore_index=True)
print("val_scv finished!")
train_csv.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/TrainInteractions.out',sep='\t',header=False,index=False)
test_csv.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/TestInteractions.out',sep='\t',header=False,index=False)
val_csv.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/ValInteractions.out',sep='\t',header=False,index=False)
print("saved!!!")
pdb.set_trace()