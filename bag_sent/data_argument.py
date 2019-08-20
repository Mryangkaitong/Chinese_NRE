# -*- coding: utf-8 -*-
#https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py
#https://github.com/tedljw/data_augment
#数据增强，shuffle或drop
import os
import sys
import time
import numpy as np
import pandas as pd 

root_dir = "./data"



def view_bar(num_cur, total):
    ratio = float(num_cur) / total
    percentage = int(ratio * 100)
    r = '\r\n[%s%s]%d%%' % (">"*percentage, " "*(100-percentage), percentage )
    sys.stdout.write(r)
    sys.stdout.flush()

def shuffle(d,e1,e2,times = 2):
    d = d.split()
    len_ = len(d)
    e1_index = d.index(e1)
    e2_index = d.index(e2)
    for i in range(times):
        index = np.random.choice(len_, 2)
        while index[0]==e1_index or index[0]==e2_index or index[1]==e1_index or index[1]==e2_index:
            index = np.random.choice(len_, 2)
        d[index[0]],d[index[1]] = d[index[1]],d[index[0]]
    return ' '.join(d)

def dropout(d,e1,e2,p=0.4):
    d = d.split()
    len_ = len(d)
    e1_index = d.index(e1)
    e2_index = d.index(e2)
    index = np.random.choice(len_, int(len_ * p))
    result = []
    for i in range(len_):
        if i== e1_index or i not in index:
            result.append(d[i])
    return ' '.join(result)


label = []
with open(os.path.join(root_dir,'sent_relation_train.txt'), 'r') as fr:
    for line in fr:
        sent_id, types = line.strip().split('\t')
        label.append(types)
i = 0
total = len(label)
argument_label_set = ['7','5','2','3','29','23','26','21','20','6','27','14','8','24','28','22','15','25','9']

print('************************开始*****************************************') 
with open(os.path.join(root_dir, 'sent_train.txt'), 'r') as fr:
    with open(os.path.join(root_dir, 'sent_train_argument.txt'), 'w') as sent_out:
        with open(os.path.join(root_dir, 'sent_relation_train_argument.txt'), 'w') as relation_out:
            for line in fr:
                id_, en1, en2, sentence = line.strip().split('\t')
                if label[i] in argument_label_set:
                    #id_tmp =  id_+'_0'
                    #sent_relation_train =id_tmp+'\t'+label[i]+'\n'
                    #sentence_drop = dropout(sentence,en1, en2)
                    #sent_train = id_tmp+'\t'+en1+'\t'+en2+'\t'+sentence_drop+'\n'
                    
                    #sent_out.write(sent_train)
                    #relation_out.write(sent_relation_train)
                    
                    
                    id_tmp =  id_+'_1'
                    sent_relation_train  = id_tmp+'\t'+label[i]+'\n'
                    sentence_shuffle = shuffle(sentence,en1, en2)
                    sent_train = id_tmp+'\t'+en1+'\t'+en2+'\t'+sentence_shuffle+'\n'
                    
                    sent_out.write(sent_train)
                    relation_out.write(sent_relation_train)
  
                sent_relation_train = id_+'\t'+label[i]+'\n'
                sent_train = line.strip()+'\n'
                sent_out.write(sent_train)
                relation_out.write(sent_relation_train)
                i = i+1
                if i % 1000 == 0 or i == total:
                    view_bar(i,total)
print('*************************结束*****************************************')           
