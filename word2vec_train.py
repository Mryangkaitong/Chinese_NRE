#coding:utf8
import os
import re
import jieba
import logging
from gensim.models import word2vec
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #训练word2vec模型时的进程数
    workers = 5
    #主目录
    baseDir = os.path.dirname(os.path.abspath(__name__))
    modelDir = baseDir+'/word2vec_model'
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    ################################################ 使用jieba进行分词，制作语料库 ################################################
    #关键词
    #jieba.set_dictionary(baseDir+'/extra_dict/dict.txt.big')
    #停用词
    stop_word_set = set()
    # 严格限制标点符号
    strict_punctuation = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
    # 简单限制标点符号
    simple_punctuation = '’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # 去除标点符号
    punctuation = simple_punctuation + strict_punctuation
#    with open(baseDir+'/extra_dict/stop_words.txt', 'r', encoding='utf-8') as sw:
#        for line in sw:
#            stop_word_set.add(line.strip('\n'))
    texts_num = 0
    #语料库保存目录
    output = open(baseDir+'/data/corpus.txt', 'w', encoding='utf-8')
    with open(baseDir+'/data/text.txt', 'r', encoding='utf-8') as content:
        for line in content:
            #去除比标点符号
            #line = re.sub('[{0}]+'.format(punctuation), '', line.strip('\n'))
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                #if word not in stop_word_set:
                output.write(word + ' ')
            texts_num += 1
            if texts_num % 1000000 == 0:
                logging.info("已完成前 %d 行的分词" % texts_num)
    output.close()
    logging.info("语料库制作完成 ！！！！！")
    ################################################ 加载处理好的语料库，训练word2vec模型 ################################################
    logging.info("训练word2vec中..................")
    sentences = word2vec.Text8Corpus(baseDir+'/data/corpus.txt')
    model = word2vec.Word2Vec(sentences,sg=1,size=300,window=5,min_count=10,negative=5,sample=1e-4,workers=workers)
    model.wv.save_word2vec_format(modelDir+'word2vec.txt',binary = False)
    logging.info("成功，结束 ！！")
if __name__ == '__main__':
    main()
