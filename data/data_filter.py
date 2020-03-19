import pandas as pd
import pdb

def parse(path):
  g = open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('/home//Desktop/Demo/CARL-master/CARL-master/data/reviews_Video_Games.json')
review_df = df[['reviewerID','asin','overall','reviewText','unixReviewTime']]
num_review = len(review_df)
print("%s %d"%('reviews:',num_review))  #reviews的条数和rating数相同
userId = review_df['reviewerID']
itemId = review_df['asin']
user = userId.unique()
num_user = len(user)
print("%s %d"%('users:',num_user))
item = itemId.unique()
num_item = len(item)
print("%s %d"%('items:',num_item))
# reviews: 1324753  users: 826767  items: 50210


# 把user和item的交互关系放到一个dict中，key：user,value：items
rating_dict = {}
for j in range(len(review_df)):
    u = userId[j]
    i = itemId[j]
    rating_dict.setdefault(u,[]).append(i)  # dict格式化，一个key对应多个value值{key:[],key1:[],.......}
    #rating_dict[u].add(i)

print("finished")


# 过滤掉交互关系少于5个item的user
non_user = []

for user,item_list in rating_dict.items():
    if len(item_list)<5:
        non_user.append(user)

# 删除reivew_df表格中过滤掉的user对应的所有记录
filter_df = review_df[~review_df['reviewerID'].isin(non_user)]
filter_user = filter_df['reviewerID'].unique()
filter_item = filter_df['asin'].unique()
print("%s %d"%('filter users:',len(filter_user)))
print("%s %d"%('filter items:',len(filter_item)))
print("%s %d"%('filter reviews:',len(filter_df)))
#users: 31027 items: 33899  reviews: 300003

# user和item的id号分别one_hot
new_user = list(range(len(filter_user)))
dict_user = dict(map(lambda x, y: [x, y], filter_user, new_user))
old_userId = filter_df['reviewerID'].values
user_list = []
for i in range(len(old_userId)):
  if old_userId[i] in dict_user.keys():
    user_list.append(dict_user[old_userId[i]])

new_item = list(range(len(filter_item)))
dict_item = dict(map(lambda x,y:[x,y],filter_item,new_item))
old_itemId = filter_df['asin'].values
item_list = []
for j in range(len(old_itemId)):
  if old_itemId[j] in dict_item.keys():
    item_list.append(dict_item[old_itemId[j]])


#保存到新的csv表格中
rating = filter_df['overall']
review_text = filter_df['reviewText']
time = filter_df['unixReviewTime']
VideoGames_df = pd.DataFrame({'user':user_list,
                          'item':item_list,
                          'rating':rating.values,
                          'time':time.values,
                          'review':review_text.values})

VideoGames_df.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/Video_Games_filter.csv',sep=',',index=False,columns=['user','item','rating','time','review'])

