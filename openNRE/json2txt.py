import os
import sys
import pandas as pd
root_path = './test_result'
test_filename = sys.argv[1]
df = pd.read_json(os.path.join(root_path, test_filename +'_pred.json'))
with open(os.path.join(root_path,'result_sent.txt'), 'w') as fw:
    for sent_id, group in df.groupby('entpair'):
        sent_id = sent_id.split('#')[0][:-2]
        if group['score'].max()-group['score'].min()<1e-5:
            relation_id=0
        else:
            relation_id_items = group[group['score']==group['score'].max()]['relation']
            i = 0
            for _,relation_id_cur in relation_id_items.items():
                if i==0:
                    relation_id = relation_id_cur
                if i>0:
                    print('#########句子 %s 最大分数有多个，这里选取第一个，除此之外结果还有：############'%sent_id)
                    print(relation_id_cur)
                i = i+1
        fw.write(sent_id + '\t' + str(relation_id) + '\n')
print('json 转化 txt完成')
