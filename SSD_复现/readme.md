# detector文件夹：
# 调用模型来检测
'''
detectior|
----|
----|
'''
>> python model_predict.py

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