# Chinese_NRE
中文关系抽取

# 数据集：

人物关系抽取比赛https://biendata.com/competition/ccks_2019_ipre/leaderboard/

备用下载链接：https://pan.baidu.com/s/1JR7L_pCIXFLLjrbRSOJw9A  提取码：obn7 

关系：
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/people_relation.png)


# 模型
这里进行了三方面的尝试

一个是使用OpenNRE，这是一个清华开发的API:https://github.com/thunlp/OpenNRE 对应于openNRE文件夹下

一个是(BGRU+2ATT)网络：https://github.com/squirrel1982/TensorFlow-NRE

最后一个是一个简单的版本对应于bag_sent文件夹下


OpenNRE其目前只实现了bag方式的单标签，没有实现多标签，且没有sent方式，不过现在好像正在开发，大家可以期待，对其感兴趣的同学可以关注：

https://github.com/thunlp/OpenNRE/tree/nrekit

所以准确来说，OpenNRE并不适用该比赛，为此，为了进一步展示bag方式（多标签）和sent这种形式，这里会结合比赛的给出的baseline的代码进行实践，补存实现pcnn,rnn,cnn(目前只有sentences)等即bag_sent文件夹下

baseline：https://github.com/ccks2019-ipre/baseline

还有就是GRU，其官方对比效果好于OpenNRE，其实可以看成是一个改进版吧

https://github.com/squirrel1982/TensorFlow-NRE

这也是本篇主要参考几篇资料，第一部分和第三部分在该比赛中效果不好，这里之所以讲主要目的就是介绍一下其使用流程，以便有需要的场合使用。相关的实践因为其只能处理单标签，这里就用sent的数据

# 开始
首先下载后数据集后，训练词向量
<pre><code>Python word2vec_train.py

</code></pre>

## OpenNRE
一 ：在OpenNRE文件夹下创建/data/people_relation/文件夹，将训练好的词向量和解压的数据放入

二 ：将txt转化成json
<pre><code> python txt2json.py

</code></pre>
三：训练
<pre><code> python train_demo.py people_relation  pcnn att

</code></pre>

四：性能
<pre><code> python draw_plot.py people_relation_pcnn_att
</code></pre>
五:预测
<pre><code> python test_demo.py people_relation  pcnn  att

</code></pre>
六：转化为txt
<pre><code> python json2txt.py people_relation_pcnn_att
</code></pre>

说明：entity文件夹下的两个脚本对应的是在转化过程中将实体对单独编码id，这个逻辑上面更通一些

## bag_sent
在该文件加下创建/data文件夹，将解压的数据和词向量放入

一：下载bert模型

因为这里看了一下bert，所以需要下载训练好的bert模型，
链接：https://pan.baidu.com/s/1ZuiOLCSluMCyVp3HhvCexw 
提取码：rhza ,
下载好后将其解压放到bag_sent/bert_model/文件夹下

二：训练
假设使用cnn 训练sent模式
<pre><code> baseline.py --encoder rnn --level sent

</code></pre>
假设使用pcnn 训练bag模式
<pre><code> baseline.py --encoder rnn --level bag

</code></pre>


四：预测
<pre><code> baseline.py --encoder rnn --level bag --mode test
</code></pre>
会在当前文件夹生成结果


# BGRU
一：将解压的数据放入origin_data目录下

二：数据预处理
<pre><code> 
python initial.py
</code></pre>
三：训练
<pre><code>
python train_GRU.py
</code></pre>
其中它会自动调用test_GRU.py验证其在dev上面的性能

四：预测结果
<pre><code> 
python predict_GRU.py 2643
</code></pre>
其中2643是加载2643模型，可以加载别的，具体看model下面有哪些即可


# 部分结果：
## OpenNRE
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/OpenNRE_rnn_one.png)

![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/OpenNRE_pcnn_att.png)

## bag_sent
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/sent_bag_result.png)

更新(线下)：

一  input+双层birnn（lstm）+attention_1
 
sent: 0.2515

二  input+单层birnn（gru）+attention_1

sent:0.2693

三  input+cnn

bag:0.2819

四  input+双层birnn+attention_2

sent:0.2698

五  input（cnn）+双层birnn+attention_2

sent:0.2088

六  双层birnn+attention_1+level_1

sent:0.254

七  双层birnn+attention_1+MASK

sent:0.272969

八  双层birnn（gru）+attention_1+drop

sent:0.276112

九  双层birnn（gru）+attention_1+drop+MASK

sent:0.264407

十  双层birnn（gru）+attention_1+drop+shuffle

sent:0.256494

十一  双层birnn（gru）+attention_1+drop+shuffle+MASK

sent:0.280276

十二  双层birnn（gru）+attention_1+shuffle+MASK

sent:0.271749


## BGRU
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/BGRU.png)
# 详细解析：
https://blog.csdn.net/weixin_42001089/article/details/95493249

# 其他探索：
一个基于bert的三元组关系抽取
https://blog.csdn.net/weixin_42001089/article/details/97657149

一个半监督的关系抽取：
https://github.com/Mryangkaitong/python-Machine-learning/tree/master/deepdive
