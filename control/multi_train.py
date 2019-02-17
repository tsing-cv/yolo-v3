#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : tsing-cv
#   Created date: 2019-02-14 18:12:26
#   Description :
#
#================================================================

import sys
sys.path.append("../")

from config import cfgs
from core.nets import yolov3
from core.data_preparation.dataset import dataset, Parser
from core.utils import utils
import tensorflow as tf 
from tensorflow.python.ops import control_flow_ops
import numpy as np


class Train():
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.dataset_batch()
        self.create_clones()
        # self.train()

    @staticmethod
    def get_update_op():
        """
        Extremely important for BatchNorm
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops is not None:
            return tf.group(*update_ops)
        return None

    @staticmethod
    def sum_gradients(clone_grads):                        
        averaged_grads = []
        for grad_and_vars in zip(*clone_grads):
            grads = []
            var = grad_and_vars[0][1]
            try:
                for g, v in grad_and_vars:
                    assert v == var
                    grads.append(g)
                grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
            except:
                # import pdb
                # pdb.set_trace()
                continue
            
            averaged_grads.append((grad, v))
            
            # tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
            # tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
            # tf.summary.scalar("variables_and_gradients_" + grad.op.name+\
            #       '_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
            # tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean',tf.reduce_mean(var))
        return averaged_grads

    @staticmethod
    def L2_Regularizer_Loss(is_freeze_batch_norm=True):
        if is_freeze_batch_norm:
            trainable_variables = [v for v in tf.trainable_variables() if 'bias' not in v.name]
        else:
            trainable_variables = [v for v in tf.trainable_variables() if 'beta' not in v.name
                                and 'gamma' not in v.name and 'bias' not in v.name]
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
        return lossL2


    def dataset_batch(self):
        tf.logging.info("Loading dataset >>>\n\tTrain dataset is in {}".format(cfgs.train_tfrecord))
        parser           = Parser(cfgs.IMAGE_H, cfgs.IMAGE_W, np.array(cfgs.ANCHORS), cfgs.NUM_CLASSES)
        trainset         = dataset(parser, cfgs.train_tfrecord, cfgs.BATCH_SIZE, shuffle=cfgs.SHUFFLE_SIZE)
        testset          = dataset(parser, cfgs.test_tfrecord , cfgs.BATCH_SIZE, shuffle=None)
        self.is_training = tf.placeholder(tf.bool)
        self.example     = tf.cond(self.is_training, lambda: trainset.get_next(), lambda: testset.get_next())

    def create_clones(self):        
        with tf.device('/cpu:0'):
            self.global_step   = tf.train.create_global_step()
            self.learning_rate = tf.train.exponential_decay(cfgs.learning_rate, 
                                                            self.global_step, 
                                                            decay_steps=cfgs.DECAY_STEPS, 
                                                            decay_rate=cfgs.DECAY_RATE, 
                                                            staircase=True)
            optimizer          = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, name='Momentum')
            tf.summary.scalar('learning_rate', self.learning_rate)
        # place clones
        losses    = 0 # for summary only
        gradients = []
        for clone_idx, gpu in enumerate(cfgs.gpus):
            reuse = clone_idx > 0
            with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
                with tf.name_scope('clone_{}'.format(clone_idx)) as clone_scope:
                    with tf.device(gpu) as clone_device:
                        self.images, *self.y_true  = self.example
                        model                      = yolov3.yolov3(cfgs.NUM_CLASSES, cfgs.ANCHORS)
                        pred_feature_map           = model.forward(self.images, is_training=self.is_training)
                        self.loss                  = model.compute_loss(pred_feature_map, self.y_true)
                        self.y_pred                = model.predict(pred_feature_map)
                        self.total_loss            = self.loss[0] / len(cfgs.gpus)
                        losses                    += self.total_loss
                        if clone_idx == 0:
                            regularization_loss = self.L2_Regularizer_Loss()
                            self.total_loss += regularization_loss
                        
                        tf.summary.scalar("Loss/Losses", losses)
                        tf.summary.scalar("Loss/Regular_loss", regularization_loss)
                        tf.summary.scalar("Loss/Total_loss", self.total_loss)
                        tf.summary.scalar("Loss/Loss_xy", self.loss[1])
                        tf.summary.scalar("Loss/Loss_wh", self.loss[2])
                        tf.summary.scalar("Loss/Loss_confs", self.loss[3])
                        tf.summary.scalar("Loss/Loss_class", self.loss[4])
                        clone_gradients = optimizer.compute_gradients(self.total_loss)
                        gradients.append(clone_gradients)
        
        # add all gradients together
        # note that the gradients do not need to be averaged, because the average operation has been done on loss.
        averaged_gradients = self.sum_gradients(gradients)
        
        apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=self.global_step)
        
        train_ops = [apply_grad_op]
        
        bn_update_op = self.get_update_op()
        if bn_update_op is not None:
            train_ops.append(bn_update_op)
        
        # moving average
        if cfgs.using_moving_average:
            tf.logging.info('\n{}\n\tusing moving average in training, with decay = {}\n{}'.format(
                '***'*20, 1-cfgs.moving_average_decay, '***'*20))
            ema = tf.train.ExponentialMovingAverage(cfgs.moving_average_decay)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([apply_grad_op]):
                # ema after updating
                train_ops.append(tf.group(ema_op))
            
        self.train_op = control_flow_ops.with_dependencies(train_ops, losses, name='train_op')
        

    def train(self):
        summary_hook = tf.train.SummarySaverHook(save_steps=20,
                                                 output_dir=cfgs.checkpoint_path,
                                                 summary_op=tf.summary.merge_all())
        logging_hook = tf.train.LoggingTensorHook(tensors={'total_loss': self.total_loss.name, 
                                                           'global_step': self.global_step.name,
                                                           'learning_rate': self.learning_rate.name,
                                                           'loss_xy': self.loss[1].name,
                                                           'loss_wh': self.loss[2].name,
                                                           'loss_confs': self.loss[3].name,
                                                           'loss_class': self.loss[4].name}, 
                                                  every_n_iter=2) 

        sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        if cfgs.gpu_memory_fraction < 0:
            sess_config.gpu_options.allow_growth = True
        elif cfgs.gpu_memory_fraction > 0:
            sess_config.gpu_options.per_process_gpu_memory_fraction = cfgs.gpu_memory_fraction
        

        with tf.train.MonitoredTrainingSession(master='', 
            is_chief=True, 
            checkpoint_dir=cfgs.checkpoint_path, 
            hooks=[tf.train.StopAtStepHook(last_step=cfgs.max_number_of_steps), 
                   # tf.train.NanTensorHook(self.total_loss),
                   summary_hook,
                   logging_hook],
            save_checkpoint_steps=1000,
            save_summaries_steps=20, 
            config=sess_config, 
            stop_grace_period_secs=120, 
            log_step_count_steps=cfgs.log_every_n_steps) as mon_sess:
            while not mon_sess.should_stop():
                _,step,y_p,y = mon_sess.run([self.train_op, self.global_step, self.y_pred, self.y_true], feed_dict={self.is_training:True})
                print (y_p)
                if step%cfgs.eval_interval == 0:
                    train_rec_value, train_prec_value = utils.evaluate(y_p,y)
                    y_pre,y_gt = mon_sess.run([self.y_pred, self.y_true], feed_dict={self.is_training:False})
                    test_rec_value, test_prec_value = utils.evaluate(y_pre,y_gt)
                    tf.logging.info("\n=======================> evaluation result <================================\n")
                    tf.logging.info("=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, train_rec_value, train_prec_value))
                    tf.logging.info("=> STEP %10d [VALID]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, test_rec_value,  test_prec_value))
                    tf.logging.info("\n=======================> evaluation result <================================\n")


if __name__ == "__main__":
    t = Train()
    sess = tf.Session()
    imgs, y = sess.run([t.images, t.y_true], feed_dict={t.is_training:True})
    print (y)
