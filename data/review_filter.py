import pandas as pd
import string
import pdb


csv_data = pd.read_csv('/home/wangshuai/Desktop/Demo/CARL-master/CARL-master/data_filter/Video_Games_filter.csv')
file = pd.DataFrame(csv_data)
review = file[['user','item','review']]
user = review['user']
item = review['item']
review_str = review['review'].values.tolist()
review_list = []
for i in range(len(review_str)):
    b = str(review_str[i])  # 把review中的每个review强制转为string格式
    a = b.translate(str.maketrans('', '', string.punctuation))    #去除每条review(string)中的标点符号
    #a.strip()  # 去除首尾空格
    review_list.append(a)


# 生成UserReviews    
user_list = user.values.tolist()
user_df = pd.DataFrame({'user':user_list,
                        'review':review_list})
user_text = user_df.groupby(by='user')['review'].sum()   # 把同一个user的所有review文本放到一起,为Series格式
# 转为dataframe格式
u_text = pd.DataFrame({'user':user_text.index,
                       'reviews':user_text.values})
# 保存为UserReviews.out
u_text.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/UserReviews.out',sep='\t',header=False,index=False)


# 生成ItemReviews
item_list = item.values.tolist()
item_df = pd.DataFrame({'item':item_list,
                       'review':review_list})
item_text = item_df.groupby(by='item')['review'].sum()
i_text = pd.DataFrame({'item':item_text.index,
                      'reviews':item_text.values})
i_text.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/ItemReviews.out',sep='\t',header=False,index=False)


# 生成WordDict
all_review = ' '.join(review_list)
all_word = all_review.split(" ")
ser = pd.Series(all_word, index = [i for i in range(len(all_word))])
word_unique = ser.unique()  # unique后的word_unique为numpy.ndarray格式
word_id = [i for i in range(len(word_unique))]
word_df = pd.DataFrame({'Word':word_unique,
                        'WordId':word_id
                        
                       })
word_df.to_csv('/home//Desktop/Demo/CARL-master/CARL-master/data_filter/WordDict.out',sep='\t',header=False,index=False)
