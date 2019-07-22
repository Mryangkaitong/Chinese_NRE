from pprint import pprint

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import utils
import sys
import tqdm

FLAGS = tf.app.flags.FLAGS

# use the legacy tensorflow 0.x model checkpoint file provided by THUNLP
# USE_LEGACY = 0


def main(_):
    # ATTENTION: change pathname before you load your model
    pathname = "./model/kbp/ATT_GRU_model-"
    predict_pathname = './out/'
    if not os.path.exists(predict_pathname):
        os.makedirs(predict_pathname)
    
    test_model_id = int(sys.argv[1])
    #none_ind = utils.get_none_id('./origin_datarelation2id.txt')
    #print("None index: ", none_ind)

    wordembedding = np.load('./data/vec.npy')

    test_y = np.load('./data/testall_y_submit.npy')
    test_word = np.load('./data/testall_word_submit.npy')
    test_pos1 = np.load('./data/testall_pos1_submit.npy')
    test_pos2 = np.load('./data/testall_pos2_submit.npy')

    test_settings = network.Settings()
    test_settings.vocab_size = len(wordembedding)
    test_settings.num_classes = len(test_y[0])
    #test_settings.big_num = len(test_y)
    print(test_word.shape)
    test_settings.big_num = 4
    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch
                predictions = sess.run(mtest.predictions, feed_dict)
                return predictions
            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)
            saver = tf.train.Saver()

            # ATTENTION: change the list to the iters you want to test !!
            # testlist = range(9025,14000,25)
            testlist = [test_model_id]
            for model_iter in testlist:
                print('当前加载的模型是:%s'%str(model_iter))
                saver.restore(sess, pathname + str(model_iter))

                time_str = datetime.datetime.now().isoformat()
                print(time_str)

                all_pred = []
                all_true = []
                for i in tqdm.tqdm(range(int(len(test_word) // int(test_settings.big_num)))):
                    pred= test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                                   test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                                   test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                                   test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    pred = np.array(pred)
                    all_pred.append(pred)
                all_pred = np.concatenate(all_pred, axis=0)
                print(all_pred.shape)
                result_filename = 'predict_'+str(model_iter)+'.txt'
                sent_id = []
                for sent_cur_id in open('./origin_data/sent_relation_test.txt'):
                    sent_cur_id =  sent_cur_id.strip()
                    sent_id.append(sent_cur_id)
                with open(os.path.join(predict_pathname,result_filename), 'w') as fw:
                    for i in range(len(sent_id)):
                        fw.write(sent_id[i] + '\t' + str(all_pred[i]) + '\n')
                #all_true = np.concatenate(all_true, axis=0)
                #accu = float(np.mean(all_accuracy))
                #all_true_inds = np.argmax(all_true, 1)
                #precision, recall, f1 = utils.evaluate_rm_neg(all_pred, all_true_inds,0)
                #print("*"*40)
                #print('Accu = %.4f, F1 = %.4f, recall = %.4f, precision = %.4f)' %(accu, f1, recall, precision))

    print('全部预测完成!!!!!!!!!!!')
if __name__ == "__main__":
    tf.app.run()
