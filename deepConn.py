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

def get_train_instance(train):
    user_input, item_input, rates = [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])
    return user_input, item_input, rates

def get_train_instance_batch_change(count, batch_size, user_input, item_input, ratings, user_reviews, item_reviews):
    users_batch, items_batch, user_input_batch, item_input_batch, labels_batch = [], [], [], [], []

    for idx in range(batch_size):
        index = (count*batch_size + idx) % len(user_input)
        users_batch.append(user_input[index])
        items_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        labels_batch.append([ratings[index]])

    return users_batch, items_batch, user_input_batch, item_input_batch, labels_batch

def cnn(user_reviews_representation_expnd, item_reviews_representation_expnd, W_u, W_i,W_u1,W_i1,drop_rate):
    conv_U = tf.nn.conv2d(user_reviews_representation_expnd,W_u,strides=[1,1,word_latent_dim,1],padding='SAME') # shape=(None, 300, 1, 50)
    conv_I = tf.nn.conv2d(item_reviews_representation_expnd,W_i,strides=[1,1,word_latent_dim,1],padding='SAME')
    convU = tf.nn.relu(conv_U)
    convI = tf.nn.relu(conv_I)
    dim = int(convU.get_shape()[1])
    pool_U = tf.nn.max_pool(convU,ksize=[1,dim,1,1],strides=[1, 1, 1, 1],padding='VALID') # (None, 1, 1, 50)
    pool_U = tf.squeeze(pool_U,[1,2])
    pool_I = tf.nn.max_pool(convI,ksize=[1,dim,1,1],strides=[1, 1, 1, 1],padding='VALID')
    pool_I = tf.squeeze(pool_I,[1,2])
    linear_U =  tf.matmul(pool_U,W_u1)
    linear_I =  tf.matmul(pool_I,W_i1)
    final_user = tf.nn.dropout(linear_U,keep_prob=drop_rate)
    final_item = tf.nn.dropout(linear_I,keep_prob=drop_rate)
    return final_user,final_item

def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    users_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    ratings = tf.placeholder(tf.float32, shape=[None, 1])
    dropout_rate = tf.placeholder(tf.float32)

    text_embedding = tf.Variable(word_embedding_mtrx, dtype=tf.float32, name="review_text_embeds")
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)

    text_mask = tf.constant([1.0] * text_embedding.get_shape()[0] + [0.0]) # shape=(805794,)

    word_embeddings = tf.concat([text_embedding, padding_embedding], 0) # TensorShape([805794, 100])
    word_embeddings = word_embeddings * tf.expand_dims(text_mask, -1) # padding_embedding和text_mask的作用？ #expand_dims(text_mask, -1),增加-1位置的维度为1
    
    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    user_reviews_representation_expnd = tf.expand_dims(user_reviews_representation, -1) # TensorShape([None, 300, 100, 1]) 因为这里users_inputs未知，所以第一个维度为None
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)
    item_reviews_representation_expnd = tf.expand_dims(item_reviews_representation, -1)

    W_u = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.03), name="W_u")
    W_i = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.03), name="W_i")
    W_u1 = tf.Variable(
        tf.truncated_normal([num_filters,latent_dim], stddev=0.3), name="W_u1")
    W_i1 = tf.Variable(
        tf.truncated_normal([num_filters,latent_dim], stddev=0.3), name="W_i1")    

    user,item = cnn(user_reviews_representation_expnd, item_reviews_representation_expnd, W_u, W_i,W_u1,W_i1,drop_rate)
    
    entity_embeds_sum = tf.concat([user,item],1)


    #FM layer
    w_entity_0 = tf.Variable(tf.zeros(1), name="entity_w_0")
    w_entity_1 = tf.Variable(tf.truncated_normal([1, latent_dim*2], stddev=0.03), name="entity_w_1")
    v_entity = tf.Variable(tf.truncated_normal([latent_dim*2, v_dim], stddev=0.03), name="entity_v")

    J_e_1 = w_entity_0 + tf.matmul(entity_embeds_sum, w_entity_1, transpose_b=True)

    entity_embeds_sum_1 = tf.expand_dims(entity_embeds_sum, -1)
    entity_embeds_sum_2 = tf.expand_dims(entity_embeds_sum, 1)
    J_e_2 = tf.reduce_sum(
        tf.reduce_sum(tf.multiply(tf.matmul(entity_embeds_sum_1, entity_embeds_sum_2), tf.matmul(v_entity, v_entity, transpose_b=True)),
                      2), 1, keep_dims=True)
    J_e_3 = tf.trace(tf.multiply(tf.matmul(entity_embeds_sum_1, entity_embeds_sum_2), tf.matmul(v_entity, v_entity, transpose_b=True)))
    J_e_total = (J_e_1 + 0.5 * (J_e_2 - tf.expand_dims(J_e_3,-1)))

    predict_rating = J_e_total
    loss1 = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))
    lamda = lambda_1 * (tf.nn.l2_loss(user) + tf.nn.l2_loss(item) + tf.nn.l2_loss(v_entity))
    loss = loss1 + lamda
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            t = time()
            loss_total = 0.0
            total_trainmse = 0.0
            count = 0.0
            for i in range(int(math.ceil(len(user_input) / float(batch_size)))):
                user_batch, item_batch, user_input_batch, item_input_batch, rates_batch = get_train_instance_batch_change(i, batch_size,user_input,
                                                                                                  item_input, rateings,
                                                                                                  user_reviews,item_reviews)
                
                _, loss_val,la,los1 = sess.run([train_step,loss,lamda,loss1],
                                       feed_dict={users : user_batch, items : item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch,
                                                  ratings: rates_batch, dropout_rate: drop_rate})
                total_trainmse += los1
                #pdb.set_trace()
                loss_total += loss_val
                count += 1.0
            t1 = time()
            mses, maes = [], []
            for i in range(len(user_input_test)):
                mses, maes = eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_tests[i], item_tests[i], user_input_test[i], item_input_test[i], rating_input_test[i], mses, maes)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t2 = time()
            print("epoch%d train time: %.3fs test time: %.3f  loss = %.3f  train_mse = %.3f testmse = %.3f  testmae = %.3f"%(e, (t1 - t), (t2 - t1), loss_total/count,total_trainmse/count, mse, mae))


def eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_tests, item_tests, user_input_tests, item_input_tests, rate_tests, rmses, maes):

    predicts = sess.run(predict_rating, feed_dict={users: user_tests, items: item_tests,users_inputs: user_input_tests, items_inputs: item_input_tests, dropout_rate: 0.5})
    row, col = predicts.shape
    for r in range(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs(predicts[r, 0] - rate_tests[r][0]))
    return rmses, maes

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    word_latent_dim = 50 # 100
    latent_dim = 30
    max_doc_length = 300
    window_size = 3
    v_dim = 50
    num_filters = 50
    learning_rate = 0.0001 # 0.0005,0.0002
    lambda_1 = 0.05
    batch_size = 100 # 200,128
    epochs = 180
    drop_rate = 0.5

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
    # get test instances
   # get test/val instances
    user_vals, item_vals, user_input_val, item_input_val, rating_input_val = get_test_list(200, valRatings, user_reviews, item_reviews)
    user_tests, item_tests, user_input_test, item_input_test, rating_input_test = get_test_list(200, testRatings, user_reviews, item_reviews)
    train_model()
