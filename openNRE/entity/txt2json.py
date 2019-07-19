import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Txt2json:
    def __init__(self,input_root_path='./data',output_root_path='./data', word_dim=300):
        self.id2relation = {}
        self.word_dim = word_dim
        self.input_root_path = input_root_path
        self.output_root_path = output_root_path
        self.entity = {}
        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)
    #生成rel2id.json
    def generate_rel2id(self,relation2id_file):
        relation2id_dict = {}
        for line in open(os.path.join(self.input_root_path, relation2id_file)):
            relation,relation_id= line.strip().split('\t')
            relation2id_dict[relation] = int(relation_id)
            self.id2relation[ relation_id] = relation
        #filew = open (os.path.join(self.output_root_path, 'rel2id.json'), 'w', encoding='utf-8')
        #json.dump(relation2id_dict,filew,cls=NpEncoder,ensure_ascii=False)
        #filew.close()
    #生成word_vec.json
    def generate_word_vec(self,word_vec_file):
        word2vec_list = []
        for line in open(os.path.join(self.input_root_path, word_vec_file)):
            content = line.strip().split()
            if len(content) != self.word_dim + 1:
                        continue
            word2vec_dict = {}
            word2vec_dict['word'] = content[0]
            word2vec_dict['vec'] = list(np.asarray(content[1:], dtype=np.float32))
            word2vec_list.append(word2vec_dict)
        filew = open (os.path.join(self.output_root_path, 'word_vec.json'), 'w', encoding='utf-8')
        json.dump(word2vec_list,filew,cls=NpEncoder,ensure_ascii=False)
        filew.close()
    #生成train.json,dev.json,test.json
    def generate_dataset(self,sent_file,relation_file,save_filename='dataset.json',label_flag=True):
        if label_flag:
            relation_dict = {}
            for line in open(os.path.join(self.input_root_path, relation_file)):
                sent_id, relation_id = line.strip().split('\t')
                relation_dict[sent_id] = relation_id
        result_list = []
        # sent_id_dict形式大概是：sent_id_dict[TEST_SENT_ID_000001] = 23#52
        sent_id_dict = {}
        for line in open(os.path.join(self.input_root_path, sent_file)):
            dataset_dict = {}
            #sentence
            sent_id, head, tail, sents = line.strip().split('\t')
            dataset_dict['sentence'] = sents
            if  head not in self.entity:
                self.entity[head] = str(len(self.entity))
            if  tail not in self.entity:
                self.entity[tail] = str(len(self.entity))
            if self.entity[head]+'#'+self.entity[tail] not in sent_id_dict:
                sent_id_dict[self.entity[head]+'#'+self.entity[tail]] = [sent_id]
            else:
                sent_id_dict[self.entity[head]+'#'+self.entity[tail]].append(sent_id)
            #head
            word_dict = {}
            word_dict['word'] = head
            word_dict['id'] = self.entity[head]
            dataset_dict['head'] = word_dict
            #tail
            word_dict = {}
            word_dict['word'] = tail
            word_dict['id'] = self.entity[tail]
            dataset_dict['tail'] = word_dict
            #relation,test没有label，统一赋值:NA
            try:
                if label_flag:
                    dataset_dict['relation'] = self.id2relation[relation_dict[sent_id]]
                else:
                    dataset_dict['relation'] = 'NA'
            except:
                #数据集异常，这里sent方式，按理来说只能有一个label
                print('########################异常数据#####################################')
                print(sent_id)
                print(relation_dict[sent_id])
                print(line)
                cur_relation_id = relation_dict[sent_id].split()[0]
                print('这里取其中一种关系作为label即: %s'%str(cur_relation_id))
                dataset_dict['relation'] = self.id2relation[cur_relation_id ]
            result_list.append(dataset_dict)
        filew = open (os.path.join(self.output_root_path, save_filename), 'w', encoding='utf-8')
        json.dump(result_list,filew,cls=NpEncoder,ensure_ascii=False)
        filew.close()
        return sent_id_dict
 
        
if __name__ == '__main__': 
    relation2id_file = 'relation2id.txt'
    word_vec_file = 'word2vec.txt'
    
    train_sent_file = 'sent_train.txt'
    train_relation_file =  'sent_relation_train.txt'
    train_save_filename = 'train.json'
    
    dev_sent_file = 'sent_dev.txt'
    dev_relation_file = 'sent_relation_dev.txt'
    dev_save_filename = 'dev.json'
    
    test_sent_file = 'sent_test.txt'
    test_relation_file = 'sent_relation_test.txt'
    test_save_filename = 'test.json'
    
    txt2json = Txt2json(input_root_path='./data',output_root_path='./data', word_dim=300)
    print('生成rel2id.json')
    txt2json.generate_rel2id(relation2id_file)
    #print('生成word_vec.json')
    #txt2json.generate_word_vec(word_vec_file)
    print('生成train.json')
    _ = txt2json.generate_dataset(train_sent_file,train_relation_file,train_save_filename)
    print('生成dev.json')
    _ = txt2json.generate_dataset(dev_sent_file,dev_relation_file,dev_save_filename)
    print('生成test.json')
    sent_id_dict = txt2json.generate_dataset(test_sent_file,test_relation_file,test_save_filename,False)
    filew = open (os.path.join('./data', 'sent_id_dict.json'), 'w', encoding='utf-8')
    json.dump(sent_id_dict,filew,ensure_ascii=False)
    filew.close()
    #print('保存实体id')
    #filew = open (os.path.join('./data', 'entity_id.json'), 'w', encoding='utf-8')
    #json.dump(txt2json.entity,filew,ensure_ascii=False)
    #filew.close()
    print('全部完成!!!!')
