import numpy as np
import os


num_relation = 35


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init():
    print('reading word embedding data...')
    vec = []
    word2id = {}
    for content in open('./origin_data/word2vec.txt', encoding='utf-8'):
        content = content.strip().split()
        if len(content) != 300 + 1:
            continue
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [float(i) for i in content]
        vec.append(content)
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    dim = 300
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    
    # length of sentence is 70
    fixlen = 70
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60

    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector
    
   
    relation_dict = {}
    for line in open('./origin_data/sent_relation_train.txt'):
        sent_id, relation_id = line.strip().split('\t')
        relation_dict[sent_id] = relation_id
    
    print('reading train data...')
    f = open('./origin_data/sent_train.txt', 'r', encoding='utf-8')
     
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        # get entity name
        en1 = content[1]
        en2 = content[2]
        
        try:
            relation = int(relation_dict[content[0]])
        except:
            relation = int(relation_dict[content[0]][0])
        
        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        label_tag = 0
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(num_relation)]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation
            label_tag = 0
            label = [0 for i in range(num_relation)]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[3].split()

        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                ps = sentence[i].split('_')
                avg_vec = np.zeros(dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        avg_vec += vec[word2id[p]]
                if c > 0:
                    avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word

        train_sen[tup][label_tag].append(output)
    
    

    print('reading dev data ...')
    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    
    relation_dict = {}
    for line in open('./origin_data/sent_relation_dev.txt'):
        sent_id, relation_id = line.strip().split('\t')
        relation_dict[sent_id] = relation_id

    f = open('./origin_data/sent_dev.txt', 'r', encoding='utf-8')
    count = 0
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        en1 = content[1]
        en2 = content[2]
        relation = 0
        try:
            relation = int(relation_dict[content[0]])
        except:
            relation = int(relation_dict[content[0]][0])
        #这里之所以不要关系是0的，是因为在测试的时候，只测试非NA的样本
        if relation==0:
            continue       
        tup = (en1, en2, count)
        count += 1

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(num_relation)]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1

        sentence = content[3].split()

        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                ps = sentence[i].split('_')
                avg_vec = np.zeros(dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        avg_vec += vec[word2id[p]]
                if c > 0:
                    avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)

    
    
    print('reading test data ...')
    test_sen_submit = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans_submit = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    f = open('./origin_data/sent_test.txt', 'r', encoding='utf-8')
    count = 0
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        en1 = content[1]
        en2 = content[2]
        #测试集上面不知道关系，这里统一设置为1,实际上就是随便设个关系就行
        relation = 1
        tup = (en1, en2, count)
        count += 1

        if tup not in test_sen_submit:
            test_sen_submit[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(num_relation)]
            test_ans_submit[y_id] = 1
            test_ans_submit[tup] = label
        else:
            y_id = relation
            test_ans_submit[tup][y_id] = 1

        sentence = content[3].split()

        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                ps = sentence[i].split('_')
                avg_vec = np.zeros(dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        avg_vec += vec[word2id[p]]
                if c > 0:
                    avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    vec.append(avg_vec)
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen_submit[tup].append(output)

    vec = np.array(vec, dtype=np.float32)


    train_x = []
    train_y = []
    test_x = []
    test_y = []
    test_x_submit = []
    test_y_submit = []

    print('organizing train data')
    f = open('./data/train_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print('organizing dev data')
    f = open('./data/test_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()
    
    
    
    print('organizing test data')
    f = open('./data/test_q&a_submit.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen_submit:
        test_x_submit.append(test_sen_submit[i])
        test_y_submit.append(test_ans_submit[i])
        tempstr = ''
        for j in range(len(test_ans_submit[i])):
            if test_ans_submit[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x_submit = np.array(test_x_submit)
    test_y_submit = np.array(test_y_submit)

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/testall_x.npy', test_x)
    np.save('./data/testall_y.npy', test_y)
    np.save('./data/testall_x_submit.npy', test_x_submit)
    np.save('./data/testall_y_submit.npy', test_y_submit)

    # get test data for P@N evaluation, in which only entity pairs with more than 1 sentence exist
    # print('get test data for p@n test')
    #
    # pone_test_x = []
    # pone_test_y = []
    #
    # ptwo_test_x = []
    # ptwo_test_y = []
    #
    # pall_test_x = []
    # pall_test_y = []
    #
    # for i in range(len(test_x)):
    #     if len(test_x[i]) > 1:
    #
    #         pall_test_x.append(test_x[i])
    #         pall_test_y.append(test_y[i])
    #
    #         onetest = []
    #         temp = np.random.randint(len(test_x[i]))
    #         onetest.append(test_x[i][temp])
    #         pone_test_x.append(onetest)
    #         pone_test_y.append(test_y[i])
    #
    #         twotest = []
    #         temp1 = np.random.randint(len(test_x[i]))
    #         temp2 = np.random.randint(len(test_x[i]))
    #         while temp1 == temp2:
    #             temp2 = np.random.randint(len(test_x[i]))
    #         twotest.append(test_x[i][temp1])
    #         twotest.append(test_x[i][temp2])
    #         ptwo_test_x.append(twotest)
    #         ptwo_test_y.append(test_y[i])
    #
    # pone_test_x = np.array(pone_test_x)
    # pone_test_y = np.array(pone_test_y)
    # ptwo_test_x = np.array(ptwo_test_x)
    # ptwo_test_y = np.array(ptwo_test_y)
    # pall_test_x = np.array(pall_test_x)
    # pall_test_y = np.array(pall_test_y)
    #
    # np.save('./data/pone_test_x.npy', pone_test_x)
    # np.save('./data/pone_test_y.npy', pone_test_y)
    # np.save('./data/ptwo_test_x.npy', ptwo_test_x)
    # np.save('./data/ptwo_test_y.npy', ptwo_test_y)
    # np.save('./data/pall_test_x.npy', pall_test_x)
    # np.save('./data/pall_test_y.npy', pall_test_y)


def seperate():
    print('reading training data')
    x_train = np.load('./data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./data/train_word.npy', train_word)
    np.save('./data/train_pos1.npy', train_pos1)
    np.save('./data/train_pos2.npy', train_pos2)

    # print('reading p-one test data')
    # x_test = np.load('./data/pone_test_x.npy')
    # print('seperating p-one test data')
    # test_word = []
    # test_pos1 = []
    # test_pos2 = []
    #
    # for i in range(len(x_test)):
    #     word = []
    #     pos1 = []
    #     pos2 = []
    #     for j in x_test[i]:
    #         temp_word = []
    #         temp_pos1 = []
    #         temp_pos2 = []
    #         for k in j:
    #             temp_word.append(k[0])
    #             temp_pos1.append(k[1])
    #             temp_pos2.append(k[2])
    #         word.append(temp_word)
    #         pos1.append(temp_pos1)
    #         pos2.append(temp_pos2)
    #     test_word.append(word)
    #     test_pos1.append(pos1)
    #     test_pos2.append(pos2)
    #
    # test_word = np.array(test_word)
    # test_pos1 = np.array(test_pos1)
    # test_pos2 = np.array(test_pos2)
    # np.save('./data/pone_test_word.npy', test_word)
    # np.save('./data/pone_test_pos1.npy', test_pos1)
    # np.save('./data/pone_test_pos2.npy', test_pos2)
    #
    # print('reading p-two test data')
    # x_test = np.load('./data/ptwo_test_x.npy')
    # print('seperating p-two test data')
    # test_word = []
    # test_pos1 = []
    # test_pos2 = []
    #
    # for i in range(len(x_test)):
    #     word = []
    #     pos1 = []
    #     pos2 = []
    #     for j in x_test[i]:
    #         temp_word = []
    #         temp_pos1 = []
    #         temp_pos2 = []
    #         for k in j:
    #             temp_word.append(k[0])
    #             temp_pos1.append(k[1])
    #             temp_pos2.append(k[2])
    #         word.append(temp_word)
    #         pos1.append(temp_pos1)
    #         pos2.append(temp_pos2)
    #     test_word.append(word)
    #     test_pos1.append(pos1)
    #     test_pos2.append(pos2)
    #
    # test_word = np.array(test_word)
    # test_pos1 = np.array(test_pos1)
    # test_pos2 = np.array(test_pos2)
    # np.save('./data/ptwo_test_word.npy', test_word)
    # np.save('./data/ptwo_test_pos1.npy', test_pos1)
    # np.save('./data/ptwo_test_pos2.npy', test_pos2)
    #
    # print('reading p-all test data')
    # x_test = np.load('./data/pall_test_x.npy')
    # print('seperating p-all test data')
    # test_word = []
    # test_pos1 = []
    # test_pos2 = []
    #
    # for i in range(len(x_test)):
    #     word = []
    #     pos1 = []
    #     pos2 = []
    #     for j in x_test[i]:
    #         temp_word = []
    #         temp_pos1 = []
    #         temp_pos2 = []
    #         for k in j:
    #             temp_word.append(k[0])
    #             temp_pos1.append(k[1])
    #             temp_pos2.append(k[2])
    #         word.append(temp_word)
    #         pos1.append(temp_pos1)
    #         pos2.append(temp_pos2)
    #     test_word.append(word)
    #     test_pos1.append(pos1)
    #     test_pos2.append(pos2)
    #
    # test_word = np.array(test_word)
    # test_pos1 = np.array(test_pos1)
    # test_pos2 = np.array(test_pos2)
    # np.save('./data/pall_test_word.npy', test_word)
    # np.save('./data/pall_test_pos1.npy', test_pos1)
    # np.save('./data/pall_test_pos2.npy', test_pos2)
    print('seperating dev all data')
    x_test = np.load('./data/testall_x.npy')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/testall_word.npy', test_word)
    np.save('./data/testall_pos1.npy', test_pos1)
    np.save('./data/testall_pos2.npy', test_pos2)

    
    
    
    print('seperating test all data')
    x_test_submit = np.load('./data/testall_x_submit.npy')
    test_word_submit = []
    test_pos1_submit = []
    test_pos2_submit = []

    for i in range(len(x_test_submit)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test_submit[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word_submit.append(word)
        test_pos1_submit.append(pos1)
        test_pos2_submit.append(pos2)

    test_word_submit = np.array(test_word_submit)
    test_pos1_submit = np.array(test_pos1_submit)
    test_pos2_submit = np.array(test_pos2_submit)

    np.save('./data/testall_word_submit.npy', test_word_submit)
    np.save('./data/testall_pos1_submit.npy', test_pos1_submit)
    np.save('./data/testall_pos2_submit.npy', test_pos2_submit)
    
# def getsmall():
#     print('reading training data')
#     word = np.load('./data/train_word.npy')
#     pos1 = np.load('./data/train_pos1.npy')
#     pos2 = np.load('./data/train_pos2.npy')
#     y = np.load('./data/train_y.npy')
#
#     new_word = []
#     new_pos1 = []
#     new_pos2 = []
#     new_y = []
#
#     # we slice some big batch in train data into small batches in case of running out of memory
#     print('get small training data')
#     for i in range(len(word)):
#         length = len(word[i])
#         if length <= 1000:
#             new_word.append(word[i])
#             new_pos1.append(pos1[i])
#             new_pos2.append(pos2[i])
#             new_y.append(y[i])
#
#         if 1000 < length < 2000:
#             new_word.append(word[i][:1000])
#             new_word.append(word[i][1000:])
#
#             new_pos1.append(pos1[i][:1000])
#             new_pos1.append(pos1[i][1000:])
#
#             new_pos2.append(pos2[i][:1000])
#             new_pos2.append(pos2[i][1000:])
#
#             new_y.append(y[i])
#             new_y.append(y[i])
#
#         if 2000 < length < 3000:
#             new_word.append(word[i][:1000])
#             new_word.append(word[i][1000:2000])
#             new_word.append(word[i][2000:])
#
#             new_pos1.append(pos1[i][:1000])
#             new_pos1.append(pos1[i][1000:2000])
#             new_pos1.append(pos1[i][2000:])
#
#             new_pos2.append(pos2[i][:1000])
#             new_pos2.append(pos2[i][1000:2000])
#             new_pos2.append(pos2[i][2000:])
#
#             new_y.append(y[i])
#             new_y.append(y[i])
#             new_y.append(y[i])
#
#         if 3000 < length < 4000:
#             new_word.append(word[i][:1000])
#             new_word.append(word[i][1000:2000])
#             new_word.append(word[i][2000:3000])
#             new_word.append(word[i][3000:])
#
#             new_pos1.append(pos1[i][:1000])
#             new_pos1.append(pos1[i][1000:2000])
#             new_pos1.append(pos1[i][2000:3000])
#             new_pos1.append(pos1[i][3000:])
#
#             new_pos2.append(pos2[i][:1000])
#             new_pos2.append(pos2[i][1000:2000])
#             new_pos2.append(pos2[i][2000:3000])
#             new_pos2.append(pos2[i][3000:])
#
#             new_y.append(y[i])
#             new_y.append(y[i])
#             new_y.append(y[i])
#             new_y.append(y[i])
#
#         if length > 4000:
#             new_word.append(word[i][:1000])
#             new_word.append(word[i][1000:2000])
#             new_word.append(word[i][2000:3000])
#             new_word.append(word[i][3000:4000])
#             new_word.append(word[i][4000:])
#
#             new_pos1.append(pos1[i][:1000])
#             new_pos1.append(pos1[i][1000:2000])
#             new_pos1.append(pos1[i][2000:3000])
#             new_pos1.append(pos1[i][3000:4000])
#             new_pos1.append(pos1[i][4000:])
#
#             new_pos2.append(pos2[i][:1000])
#             new_pos2.append(pos2[i][1000:2000])
#             new_pos2.append(pos2[i][2000:3000])
#             new_pos2.append(pos2[i][3000:4000])
#             new_pos2.append(pos2[i][4000:])
#
#             new_y.append(y[i])
#             new_y.append(y[i])
#             new_y.append(y[i])
#             new_y.append(y[i])
#             new_y.append(y[i])
#
#     new_word = np.array(new_word)
#     new_pos1 = np.array(new_pos1)
#     new_pos2 = np.array(new_pos2)
#     new_y = np.array(new_y)
#
#     np.save('./data/small_word.npy', new_word)
#     np.save('./data/small_pos1.npy', new_pos1)
#     np.save('./data/small_pos2.npy', new_pos2)
#     np.save('./data/small_y.npy', new_y)


# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/testall_y.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./data/allans.npy', allans)

def getans():
    test_y = np.load('./data/testall_y_submit.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./data/allans_submit.npy', allans)
# def get_metadata():
#     fwrite = open('./data/metadata.tsv', 'w', encoding='utf-8')
#     f = open('./origin_data/vec.txt', encoding='utf-8')
#     f.readline()
#     while True:
#         content = f.readline().strip()
#         if content == '':
#             break
#         name = content.split()[0]
#         fwrite.write(name + '\n')
#     f.close()
#     fwrite.close()


init()
seperate()
# getsmall()
getans()
# get_metadata()
