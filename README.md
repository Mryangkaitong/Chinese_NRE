# Chinese_NRE
中文关系抽取

# 数据集：

人物关系抽取比赛https://biendata.com/competition/ccks_2019_ipre/leaderboard/

备用下载链接：https://pan.baidu.com/s/1JR7L_pCIXFLLjrbRSOJw9A  提取码：obn7 

关系：
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/people_relation.png)


# 模型
这里进行了两方面的尝试

一个是使用OpenNRE，这是一个清华开发的API:https://github.com/thunlp/OpenNRE 对应于openNRE文件夹下

一个是一个简单的版本对应于bag_sent文件夹下


在OpenNRE上面跑的结果很差（具体原因不知道怎么回事？难道是转化数据格式有问题？），后续参考比赛demo和OpenNRE又写了一个简单的版本

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

# 部分结果：
## OpenNRE
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/OpenNRE_rnn_one.png)

![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/OpenNRE_pcnn_att.png)

## bag_sent
![image](https://github.com/Mryangkaitong/Chinese_NRE/blob/master/photo/sent_bag_result.png)

# 详细解析：
https://blog.csdn.net/weixin_42001089/article/details/95493249
