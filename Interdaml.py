# encoding: utf-8
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution() # 忽略占位符
import numpy as np
#from auxiliaryTools.ExtractData import Dataset
#from auxiliaryTools.GetTest import get_test_list
from ExtractData import Dataset
from GetTest import get_test_list
from time import time
import math, os
import pdb


def ini_word_embed(num_words, latent_dim):
    word_embeds = np.random.rand(num_words, latent_dim)
    return word_embeds

def word2vec_word_embed(num_words, latent_dim, path, word_id_dict):
    word2vect_embed_mtrx = np.zeros((num_words, latent_dim))
    with open(path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            row_id = word_id_dict.get(arr[0])
            vect = arr[1].strip().split(" ")
            for i in range(len(vect)):
                word2vect_embed_mtrx[row_id, i] = float(vect[i])  # 将预训练的word2vec向量值每一维度放到矩阵中，每行表示一个word的vector表示
            line = f.readline()

    return word2vect_embed_mtrx

def get_train_instance(train):
    user_input, item_input, rates = [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])  # 输入的train为train的dok_matrix
    return user_input, item_input, rates

def get_train_instance_batch_change(count, batch_size, user_input, item_input, ratings, user_reviews, item_reviews):
    users_batch, items_batch, user_input_batch, item_input_batch, labels_batch = [], [], [], [], []

    for idx in range(batch_size):
        index = (count*batch_size + idx) % len(user_input) # user_input为所有的user input,len(user_input)>等于%的左边，用%取左边的余数，就是为了得到左边某个batch中某个位置的user在实际user-input中的序号
        
        users_batch.append(user_input[index])
        items_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        labels_batch.append([ratings[index]])

    return users_batch, items_batch, user_input_batch, item_input_batch, labels_batch


def L_atten(user_reviews_representation_expnd, item_reviews_representation_expnd, W_u, W_i):
    convU = tf.nn.conv2d(user_reviews_representation_expnd,W_u,strides=[1,1,word_latent_dim,1],padding='SAME')
    convI = tf.nn.conv2d(item_reviews_representation_expnd,W_i,strides=[1,1,word_latent_dim,1],padding='SAME')
    # convU shape=(?, 300, 1, 1)
    u_scores = tf.nn.sigmoid(convU)
    i_scores = tf.nn.sigmoid(convI)
    return u_scores,i_scores

def Conv(filters,user,item,W_u,W_i):
    convU = tf.nn.conv2d(user,W_u,strides=[1,1,word_latent_dim,1],padding='SAME')
    convI = tf.nn.conv2d(item,W_i,strides=[1,1,word_latent_dim,1],padding='SAME')
    return convU,convI


def EuclideanDistances(A, B):
    # vecProd = A * BT
    vecProd = tf.matmul(A,B,transpose_b=True) # v
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = tf.reduce_sum(SqA, axis=2,keepdims=True) # (None, 300, 1)
    sumSqAEx = tf.tile(sumSqA, (1,1, vecProd.shape[2])) # (None, 300, 300)
    # print(sumSqAEx)
 
    SqB = B**2
    sumSqB = tf.reduce_sum(SqB,axis=2) # (None, 300)
    sumSqBEx = tf.tile(tf.expand_dims(sumSqB,1), (1,vecProd.shape[1], 1)) # (None, 300, 300)
    SqED = sumSqBEx + sumSqAEx - 2*vecProd  
    ED = tf.sqrt(SqED) 
    return ED


def Mutual(rand_matrix,filters,user,item):
    sec_dim = int(user.get_shape()[1])
    tmphU = tf.reshape(user,[-1,filters])
    hU_mul_rand = tf.reshape(tf.matmul(tmphU,rand_matrix),[-1,sec_dim,filters])
    f = tf.matmul(hU_mul_rand,item,transpose_b=True)# 三个维度的矩阵乘积是 对应位置的二维矩阵乘积
    f = tf.expand_dims(f,-1)
    att1 = tf.tanh(f)
    pool_user = tf.reduce_mean(att1,2)
    pool_item = tf.reduce_mean(att1,1)

    user_flat = tf.squeeze(pool_user, -1)
    item_flat = tf.squeeze(pool_item, -1)
    weight_user = tf.nn.softmax(user_flat) # TensorShape([None, 300])
    weight_item = tf.nn.softmax(item_flat)
    weight_user_exp = tf.expand_dims(weight_user, -1)  # [batchsize,300,1]
    weight_item_exp = tf.expand_dims(weight_item, -1)
    return weight_user_exp,weight_item_exp

def Local(user,item,W_u,W_i,W_u1,W_i1):
    convU = tf.nn.conv2d(user,W_u,strides=[1,1,1,1],padding='SAME') # shape=(None, 300, 50, 1)
    convI = tf.nn.conv2d(item,W_i,strides=[1,1,1,1],padding='SAME')

    convU1 = tf.nn.conv2d(convU,W_u1,strides=[1,1,1,1],padding='VALID') # shape=(None, 298, 1, 50)
    convI1 = tf.nn.conv2d(convI,W_i1,strides=[1,1,1,1],padding='VALID')

    sec_dim = int(convU1.get_shape()[1])
    U_acti = tf.nn.relu(convU1)
    I_acti = tf.nn.relu(convI1)
    hU = tf.nn.avg_pool(convU1, ksize=[1, sec_dim, 1, 1], strides=[1, 1, 1, 1],
        padding='VALID')   # ksize=[1, height, width, 1] oU:(batchsize, 1, 1, 50) hU:[batch_size,1,1,50]
    hI = tf.nn.avg_pool(convI1,ksize=[1,sec_dim,1,1],strides=[1,1,1,1],
        padding='VALID')
    hU = tf.squeeze(hU,[1,2])  # [None,50]
    hI = tf.squeeze(hI,[1,2])
    return hU,hI

def MultiLayerPerceptron(fz,W1,b1,W2,b2,hT):
    mlp1 = tf.matmul(fz,W1)+b1
    #mlp2 = tf.nn.batch_normalization(mlp1)
    mlp3 = tf.nn.relu(mlp1)
    mlp4 = tf.nn.dropout(mlp3, keep_prob=0.5)

    mlp5 = tf.matmul(mlp4,W2)+b2
    #mlp6 = tf.nn.batch_normalization(mlp5)
    mlp7 = tf.nn.relu(mlp5)
    mlp8 = tf.nn.dropout(mlp7,keep_prob=0.5)
    final = tf.matmul(mlp8,hT,transpose_b=True)
    return final



def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    users_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    ratings = tf.placeholder(tf.float32, shape=[None, 1])
    dropout_rate = tf.placeholder(tf.float32)

    text_embedding = tf.Variable(word_embedding_mtrx, dtype=tf.float32, name="review_text_embeds")
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)

    text_mask = tf.constant([1.0] * text_embedding.get_shape()[0] + [0.0])

    word_embeddings = tf.concat([text_embedding, padding_embedding], 0) # TensorShape([805794, 100])
    word_embeddings = word_embeddings * tf.expand_dims(text_mask, -1) # padding_embedding和text_mask的作用？ #expand_dims(text_mask, -1),增加-1位置的维度为1
    
    user_entity_embedding = tf.Variable(tf.random_normal([num_users, latent_dim], mean=0, stddev=0.02), name="user_entity_embeddings")
    item_entity_embedding = tf.Variable(tf.random_normal([num_items, latent_dim], mean=0, stddev=0.02), name="item_entity_embeddings")
    user_bias = tf.Variable(tf.random_normal([num_users, 1], mean=0, stddev=0.02), name="review_user_bias")
    item_bias = tf.Variable(tf.random_normal([num_items, 1], mean=0, stddev=0.02), name="review_item_bias")
    
    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)

    user_entity_embeds = tf.nn.embedding_lookup(user_entity_embedding, users)  
    item_entity_embeds = tf.nn.embedding_lookup(item_entity_embedding, items)  

    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    user_reviews_representation_expnd = tf.expand_dims(user_reviews_representation, -1) # TensorShape([None, 300, 100, 1]) 因为这里users_inputs未知，所以第一个维度为None
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)
    item_reviews_representation_expnd = tf.expand_dims(item_reviews_representation, -1)

    # L_attn layers
    W_u = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, 1], stddev=0.3), name="Latten_W_u")
    W_i = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, 1], stddev=0.3), name="Latten_W_i")
    # user_reviews_represention_expand:TensorShape([None, 300, 100, 1])    
    u_scores,i_scores = L_atten(user_reviews_representation_expnd, item_reviews_representation_expnd, W_u, W_i)
    u_scores = tf.squeeze(u_scores,-1)  # u_scores[None,300,1]
    i_scores = tf.squeeze(i_scores,-1) 
    user_reviews_representation_expnd = tf.squeeze(user_reviews_representation_expnd,-1)
    item_reviews_representation_expnd = tf.squeeze(item_reviews_representation_expnd,-1)
    att_user = tf.multiply(user_reviews_representation_expnd,u_scores)  # ([None, 300, 100])
    att_item = tf.multiply(item_reviews_representation_expnd,i_scores) 

    # Convolution Operation
    W_u1 = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="Conv_W_u")
    W_i1 = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="Conv_W_i")
    att_user = tf.expand_dims(att_user,-1)  
    att_item = tf.expand_dims(att_item,-1)    
    Conv_user,Conv_item = Conv(num_filters,att_user,att_item,W_u1,W_i1) # Conv_user:shape=(None, 300, 1, 50)
    Conv_user = tf.squeeze(Conv_user,2) # [None,300,50]
    Conv_item = tf.squeeze(Conv_item,2)

    # Mutual attention layer
    euclidean = EuclideanDistances(Conv_user,Conv_item) # (None, 300, 300)
    euclidean = 1/(1+euclidean)
    eu_user = tf.reduce_mean(euclidean,axis=2)  # 按行求和 
    eu_item = tf.reduce_mean(euclidean,axis=1)  # 按列求和 原文reduce_sum(),用FM输出rating时reduce_mean()输出结果才正确
    eu_user = tf.expand_dims(eu_user,-1) # (None, 300, 1)
    eu_item = tf.expand_dims(eu_item,-1)
    Mul_user = Conv_user*eu_user
    Mul_item = Conv_item*eu_item
  


    # rand_matrix = tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.3), name="review_rand_matrix")
    # Mul_user_score,Mul_item_score = Mutual(rand_matrix,num_filters,Conv_user,Conv_item)
    # Mul_user = Conv_user*Mul_user_score
    # Mul_item = Conv_item*Mul_item_score

    # Local pooling layer
    dim = int(Mul_user.get_shape()[2])
    # W_u2 = tf.Variable(
    #     tf.truncated_normal([window_size, dim, 1, 1], stddev=0.3), name="Conv_W_u2")
    # W_i2 = tf.Variable(
    #     tf.truncated_normal([window_size, dim, 1, 1], stddev=0.3), name="Conv_W_u2")

    W_u2 = tf.Variable(
        tf.truncated_normal([window_size, 1, 1, 1], stddev=0.3), name="Conv_W_u2")
    W_i2 = tf.Variable(
        tf.truncated_normal([window_size, 1, 1, 1], stddev=0.3), name="Conv_W_u2")

    # W_u3 = tf.Variable(
    #     tf.truncated_normal([window_size, 1, 1, num_filters], stddev=0.3), name="Conv_W_u3")
    # W_i3 = tf.Variable(
    #     tf.truncated_normal([window_size, 1, 1, num_filters], stddev=0.3), name="Conv_W_u3")

    W_u3 = tf.Variable(
        tf.truncated_normal([window_size, dim, 1, num_filters], stddev=0.3), name="Conv_W_u3")
    W_i3 = tf.Variable(
        tf.truncated_normal([window_size, dim, 1, num_filters], stddev=0.3), name="Conv_W_u3")
    
    Mul_user = tf.expand_dims(Mul_user,-1)
    Mul_item = tf.expand_dims(Mul_item,-1)
    user,item = Local(Mul_user,Mul_item,W_u2,W_i2,W_u3,W_i3)

    
    # MLP layer
    W_mlp_u = tf.Variable(tf.random_normal([num_filters, latent_dim], mean=0, stddev=0.2), name="review_W_mlp_u")
    #b_mlp_u = tf.Variable(tf.constant(0., shape=[batch_size,1]), name="review_b_mlp_u")
    b_mlp_u = tf.Variable(tf.constant(0., shape=[latent_dim]), name="review_b_mlp_u")
    
    W_mlp_i = tf.Variable(tf.random_normal([num_filters, latent_dim], mean=0, stddev=0.2), name="review_W_mlp_i")
    #b_mlp_i = tf.Variable(tf.constant(0., shape=[batch_size,1]), name="review_b_mlp_i")
    b_mlp_i = tf.Variable(tf.constant(0., shape=[latent_dim]), name="review_b_mlp_i")
    
    a = tf.matmul(user, W_mlp_u)
 
    user_embeds = tf.nn.relu(tf.matmul(user, W_mlp_u) + b_mlp_u) # [batch_size,latent_dim]
    item_embeds = tf.nn.relu(tf.matmul(item, W_mlp_i) + b_mlp_i)

    # Feature Interaction 
    u_map = tf.Variable(tf.truncated_normal([latent_dim,latent_dim],stddev=0.3),name="Inter_u")
    i_map = tf.Variable(tf.truncated_normal([latent_dim,latent_dim],stddev=0.3),name="Inter_i")
    Interaction_u = tf.matmul(user_entity_embeds,u_map)
    Interaction_i = tf.matmul(item_entity_embeds,i_map)

    # FM layer
    final_user = user_embeds+Interaction_u
    final_item = item_embeds+Interaction_i

    embeds_sum = tf.concat([final_user, final_item], 1, name="concat_embed") # [batchsize,2*latent_dim]
    w_0 = tf.Variable(tf.zeros(1), name="review_w_0")
    w_1 = tf.Variable(tf.truncated_normal([1, latent_dim*2], stddev=0.3), name="review_w_1")
    v = tf.Variable(tf.truncated_normal([latent_dim * 2, v_dim], stddev=0.3), name="review_v") # [2*latent_dim,v_dim]

    J_1 = w_0 + tf.matmul(embeds_sum, w_1, transpose_b=True) # FM的线性部分
    
    embeds_sum_1 = tf.expand_dims(embeds_sum, -1) # [batchsize,2*latent_dim,1]
    embeds_sum_2 = tf.expand_dims(embeds_sum, 1)  # [batchsize,1,2*latent_dim]

    J_2 = tf.reduce_sum(
        tf.reduce_sum(tf.multiply(tf.matmul(embeds_sum_1, embeds_sum_2), tf.matmul(v, v, transpose_b=True)),
                      2), 1, keep_dims=True)  # [200,1]
    J_3 = tf.trace(tf.multiply(tf.matmul(embeds_sum_1, embeds_sum_2), tf.matmul(v, v, transpose_b=True))) # tf.multiply() [batchsize,2*latent_dim,2*latent_dim]
    fz = 0.5*(J_2-tf.expand_dims(J_3,-1)) # [batchsize,1]
    predict_rating = J_1+fz+user_bs+item_bs
    loss1 = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))
    lamda = lambda_1*(tf.nn.l2_loss(final_user)+tf.nn.l2_loss(final_item)+tf.nn.l2_loss(v))
    loss = loss1+lamda
    #loss += lambda_1*(tf.nn.l2_loss(W_u)+tf.nn.l2_loss(W_i)+tf.nn.l2_loss(W_u1)+tf.nn.l2_loss(W_i1)+tf.nn.l2_loss(W_u2)+tf.nn.l2_loss(W_i2)+tf.nn.l2_loss(W_u3)+tf.nn.l2_loss(W_i3)+tf.nn.l2_loss(v)+tf.nn.l2_loss(user_bs)+tf.nn.l2_loss(item_bs)+tf.nn.l2_loss(user_entity_embedding)+tf.nn.l2_loss(item_entity_embedding)+tf.nn.l2_loss(w_1)+tf.nn.l2_loss(user_bs)+tf.nn.l2_loss(item_bs))
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=1)
    # # MLP
    # dim = int(fz.get_shape()[1])
    # W1 = tf.Variable(tf.truncated_normal([dim,mlp_dims[0]],stddev=0.3),name="W1")
    # bias1 = tf.Variable(tf.truncated_normal([mlp_dims[0]],stddev=0.002),name="b1")
    # W2 = tf.Variable(tf.truncated_normal([mlp_dims[0],mlp_dims[1]],stddev=0.3),name="W2")
    # bias2 = tf.Variable(tf.truncated_normal([mlp_dims[1]],stddev=0.002),name="b2")
    # hT = tf.Variable(tf.truncated_normal([1,mlp_dims[1]]))
    # mlp = MultiLayerPerceptron(fz,W1,bias1,W2,bias2,hT)
 
    # J_total = J_1 + mlp

    # #J_total = (J_1 + 0.5 * (J_2 - tf.expand_dims(J_3,-1))) # 0.5 * (J_2 - J_3)是FM的交互部分 #
    # predict_rating = J_total + user_bs + item_bs
    # loss = tf.reduce_mean(tf.squared_difference(predict_rating, ratings)) # ratings:[200,1]
    # loss += lambda_1*(tf.nn.l2_loss(W_u)+tf.nn.l2_loss(W_i)+tf.nn.l2_loss(W_u1)+tf.nn.l2_loss(W_i1)+tf.nn.l2_loss(W_u2)+tf.nn.l2_loss(W_i2)+tf.nn.l2_loss(W_u3)+tf.nn.l2_loss(W_i3)+tf.nn.l2_loss(v)+tf.nn.l2_loss(user_bs)+tf.nn.l2_loss(item_bs)+tf.nn.l2_loss(user_entity_embedding)+tf.nn.l2_loss(item_entity_embedding)+tf.nn.l2_loss(w_1)+tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(hT)+tf.nn.l2_loss(W_mlp_u)+tf.nn.l2_loss(W_mlp_i))
    # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    #saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for e in range(epochs):
            t = time()
            loss_total = 0.0
            train_msetotal = 0.0
            count = 0.0
            all_predict = []
            for i in range(int(math.ceil(len(user_input) / float(batch_size)))):   # match.ceil(x)：x的上入整数
                user_batch, item_batch, user_input_batch, item_input_batch, rates_batch = get_train_instance_batch_change(i, batch_size,user_input,
                                                                                                  item_input, rateings,
                                                                                                  user_reviews,item_reviews)
                predict,los1,la,_, loss_val, words,scores,user,item = sess.run([predict_rating,loss1,lamda,train_step, loss, word_embeddings,euclidean,eu_user,eu_item],
                                       feed_dict={users : user_batch, items : item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch,
                                                  ratings: rates_batch, dropout_rate: drop_out}) #模型中需要训练的参数都要用sess.run执行
                a = predict.squeeze(-1)
                all_predict.append(a)
                #pdb.set_trace()
                #print(loss_val)
                loss_total += loss_val
                train_msetotal += los1
                count += 1.0
            t1 = time()
            #print("epoch "+str(e)+" train_predict_mean "+str(np.mean(all_predict))+" train_predict_var "+str(np.var(all_predict)))
            #print("epoch%d loss = %.3f "%(e, loss_total/count))
            # 加载保存了的模型参数
            #saver.restore(sess,tf.train.latest_checkpoint('/home//Desktop/Demo/CARL-master/CARL-master/checkpoint/'))
            val_mses, val_maes = [], []
            predict1 = []
            for i in range(len(user_input_val)): # 遍历user_input_val中的每个batch
                eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating,  sess, user_vals[i], item_vals[i], user_input_val[i], item_input_val[i], rating_input_val[i], val_mses, val_maes,predict1)
            val_mse = np.array(val_mses).mean()
            #print("epoch "+str(e)+" val_predict_mean "+str(np.mean(predict1))+" val_predict_var "+str(np.var(predict1)))
            
          
            t2 = time()
            mses, maes = [], []
            predict2 = []
            for i in range(len(user_input_test)):
                eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_tests[i], item_tests[i], user_input_test[i], item_input_test[i], rating_input_test[i], mses, maes,predict2)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            #print("epoch "+str(e)+" val_predict_mean "+str(np.mean(predict2))+" val_predict_var "+str(np.var(predict2)))
            t3 = time()
            print("epoch%d train time: %.3fs  test time: %.3f  loss = %.3f train_mse = %.3f val_mse = %.3f test_mse = %.3f test_mae = %.3f"%(e, (t1 - t), (t3 - t2), loss_total/count, train_msetotal/count, val_mse, mse, mae))
        # 保存模型参数
        #saver.save(sess,'/home//Desktop/Demo/CARL-master/CARL-master/checkpoint/Interdaml.ckpt',global_step=1)


def eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_tests, item_tests, user_input_tests, item_input_tests, rate_tests, rmses, maes,predict_all):

    predicts = sess.run(predict_rating, feed_dict={users : user_tests, items: item_tests, users_inputs: user_input_tests, items_inputs: item_input_tests, dropout_rate: 1.0})
    row, col = predicts.shape
    for r in range(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs(predicts[r, 0] - rate_tests[r][0]))
    b = predicts.squeeze(-1)
    predict_all.extend(b)
    return rmses, maes,predict_all

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" #0
    word_latent_dim = 50 # 100
    latent_dim = 30
    max_doc_length = 300
    num_filters = 50
    window_size = 3
    v_dim = 50
    learning_rate = 0.0001 # 0.0002 0.0001
    lambda_1 = 0.005 # 0.005
    drop_out = 0.8
    batch_size = 180 # 200 180
    epochs = 180 # 180
    mlp_dims = (32,12)
    # loading data
    firTime = time()
    dataSet = Dataset(max_doc_length, "/home//Desktop/Demo/CARL-master/CARL-master/data_filter/",
                      "WordDict.out")
    word_dict, user_reviews, item_reviews, train, valRatings, testRatings = dataSet.word_id_dict, dataSet.userReview_dict, dataSet.itemReview_dict, dataSet.trainMtrx, dataSet.valRatings, dataSet.testRatings
    secTime = time()

    num_users, num_items = train.shape
    print("load data: %.3fs" % (secTime - firTime))
    print(num_users, num_items)

    #load word embeddings
    word_embedding_mtrx = ini_word_embed(len(word_dict), word_latent_dim)
    # word_embedding_mtrx = word2vec_word_embed(len(word_dict), word_latent_dim,
    #                                           "Directory of pretrained WordEmbedding.out",
    #                                           word_dict)

    print("word_dict shape", word_embedding_mtrx.shape)

    # get train instances
    user_input, item_input, rateings = get_train_instance(train)

    # get test/val instances
    user_vals, item_vals, user_input_val, item_input_val, rating_input_val = get_test_list(200, valRatings, user_reviews, item_reviews)
    user_tests, item_tests, user_input_test, item_input_test, rating_input_test = get_test_list(200, testRatings, user_reviews, item_reviews)

    #train & eval model
    train_model()
  
