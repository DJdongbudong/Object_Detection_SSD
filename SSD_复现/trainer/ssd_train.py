import tensorflow as tf
import numpy as np
from ssd import ssd
import tfr_data_process
import preprocess_img_tf
import util_tf
import matplotlib.pyplot as plt
slim = tf.contrib.slim


# max_steps = 10000
# batch_size = 16
max_steps = 1000000
batch_size = 64

num_epochs_per_decay = 2.0
num_samples_per_epoch = 2560



sd = ssd()
# global_step = tf.train.create_global_step()
#1----------------->获取所有anchor_boxes
layers_anchors = []
for i, s in enumerate(sd.feature_map_size):
    anchor_bboxes = sd.ssd_anchor_layer(sd.img_size, s,
                                          sd.anchor_sizes[i],
                                          sd.anchor_ratios[i],
                                          sd.anchor_steps[i],
                                          sd.boxes_len[i])
    layers_anchors.append(anchor_bboxes)

# 2-------------->数据加载
dataset = tfr_data_process.get_split('TFR_Data',
                                       'voc_2007_train_*.tfrecord',
                                       num_classes=21,
                                       num_samples=num_samples_per_epoch)
image, glabels, ggggbboxes = tfr_data_process.tfr_read(dataset)


image, glabels, gbboxes = \
                preprocess_img_tf.preprocess_image(image, glabels, ggggbboxes, out_shape=(300, 300))

# 2.1 resize
# image = util_tf.resize_image(image,size=(300, 300))
# #2.2 white
# image = util_tf.tf_image_whitened(image)


#3----------------> 编码网络

target_labels,target_localizations,target_scores = sd.bboxes_encode(glabels, gbboxes,layers_anchors)

batch_shape = [1] + [len(layers_anchors)] * 3 #[1,6,6,6]



r = tf.train.batch(  # 图片，中心点类别，真实框坐标，得分
                util_tf.reshape_list([image, target_labels, target_localizations, target_scores]),
                batch_size=batch_size,
                num_threads=4,
                capacity=5 * batch_size)


batch_queue = slim.prefetch_queue.prefetch_queue(
                r,
                capacity=2)


b_image, b_gclasses, b_glocalisations, b_gscores = util_tf.reshape_list(batch_queue.dequeue(), batch_shape)

#4----------------------->经过网络
pred_locations,pred_predictions,logit = sd.set_net(x = b_image)

cls_pos_loss,cls_neg_loss,loca_loss = sd.ssd_losses(logit,pred_locations,
                                              b_gclasses,b_glocalisations,b_gscores)

total_loss = tf.reduce_sum([cls_neg_loss, cls_pos_loss, loca_loss])


# global_step = tf.Variable(0)
'''
这里learning_rate 采用指数衰减法,
decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)  
         learning_rate为事先设定的初始学习率；
        
         decay_rate为衰减系数；
    
         decay_steps为衰减速度。
         global_step:全局步数
                   
而tf.train.exponential_decay函数则可以通过staircase(默认值为False,当为True时，
（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
'''
global_step = tf.train.create_global_step()
'''
num_epochs_per_decay = 2.0
num_samples_per_epoch = 17125
batch_size = 64
decay_steps = '每隔多少次更新一次lr'
decay_steps = 17125/64 * 2
'''
decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)


# 初始学习率，全局步骤，衰减速度，衰减速率
# 喂入一次 BACTH_SIZE 计为一次 global_step；每间隔decay_steps次更新一次learning_rate值
learning_rate = tf.train.exponential_decay(0.1,
                                           global_step,
                                           decay_steps,
                                           0.94,  # learning_rate_decay_factor,
                                           staircase=True,
                                           name='exponential_decay_learning_rate')


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss,global_step=global_step)




init_op = tf.initialize_all_variables()



print('开始训练')

import os
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    # 
    

    loss_array = []
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
    
    # 1.fitune after initing
    '''    
    # 1.模型路径: 必须指定到具体的模型下如：xx.ckpt.meta
    saver = tf.train.import_meta_graph(r'./checkpoint/ssd_vgg_300_weights.ckpt.meta')  
    # 2.数据路径: 必须指定到具体某个模型的数据，但创建这个路径的方法很多，
    # 比如调用最后一个保存的模型tf.train.latest_checkpoint('./checkpoint_dir')，
    # 也可以是xx.ckpt.data,并且这两个是等效的.
    saver.restore(sess, tf.train.latest_checkpoint(r'./checkpoint/')) 
    #saver.restore(sess, 'ssd_vgg_300_weights.ckpt.data-00000-of-00001') 
    '''
    # 2.fitune after initing —— 好像无法使用预训练模型，尴尬了
    '''  
    ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
    saver.restore(sess, ckpt_filename)
    '''

    # print('batch_queue', sess.run(batch_queue))
    for step in range(max_steps):


        # sess.run(optimizer)
        loss_value = sess.run(total_loss)

        loss_array.append(loss_value)
        if step%100 == 0:
            print("第%d次的误差为：%f" % (step, loss_value))
        # print('no_',sess.run(no_classes))


        if step%1000 == 0:
            MODEL_SAVER_PATH = 'result_model'
            MODEL_NAME = 'ssd_model.ckpt'
            saver.save(sess, os.path.join(MODEL_SAVER_PATH, MODEL_NAME), global_step=step)
            plt.plot([i for i in range(len(loss_array))], loss_array)
            figname = str(step) + 'fig.png' 
            plt.savefig(figname)
            #plt.show()
        if step%10000 == 0:
            plt.plot([i for i in range(len(loss_array))], loss_array)
            figname = str(step) + 'fig_wan.png' 
            plt.savefig(figname)
            pass
        if step%100000 == 0:
            plt.plot([i for i in range(len(loss_array))], loss_array)
            figname = str(step) + 'fig_swan.png' 
            plt.savefig(figname)
            pass
    coord.request_stop()
    coord.join(threads)