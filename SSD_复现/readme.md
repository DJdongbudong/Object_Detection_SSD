# detector文件夹：
# 调用模型来检测
'''
detectior|\\
----|\\
----|
'''
>> python model_predict.py

# tf_records_creater文件夹：
# 数据转换用
'''
tf_records_creater|\\
----|\\
----|
'''
Usage:
```shell
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=/tmp/pascalvoc \
    --output_name=pascalvoc \
    --output_dir=/tmp/
```
# Window shell:
```
# 规定pascalvoc，数据集文件夹路径*/*/，指定数据集名称，指定数据集输出地址
python tf_convert_data.py  --dataset_name=pascalvoc  --dataset_dir=./voc2007/  --output_name=voc_2007_train  --output_dir=./TFR_Data
```
"""

# trainer文件夹：
# 训练ssd，数据转换模仿ssd-tensorflow的数据转换即可。
'''
trainer|
----|
----|
----|
'''
>> python ssd_train.py

# 需要改进：
'''
貌似还不能使用预训练模型来继续训练，所以下一步就开始使用SSD-TensorFlow来实现，至于该code会继续跟进，查看哪里不能调用预训练模型，可恶。
'''
