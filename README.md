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








