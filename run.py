import re
import gat_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import os
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

word_dim = 100
edge_word_dim = 50
max_length = 128
class_num = 5
batch_size = 64
learning_rate = 0.0001

def train(num):
    Epoch = 150
    if class_num == 2:
        train_sentence_ids_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_sentence_ids.npy'
        test_sentence_ids_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_sentence_ids.npy'

        train_sentence_ids = np.load(train_sentence_ids_path, allow_pickle=True)
        test_sentence_ids = np.load(test_sentence_ids_path, allow_pickle=True)

        train_label_ids_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_label_ids.npy'
        test_label_ids_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_label_ids.npy'

        train_label_ids = np.load(train_label_ids_path, allow_pickle=True)
        test_label_ids = np.load(test_label_ids_path, allow_pickle=True)

        train_entity_index_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_entity_index.npy'
        test_entity_index_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_entity_index.npy'

        train_entity_index = np.load(train_entity_index_path, allow_pickle=True)
        test_entity_index = np.load(test_entity_index_path, allow_pickle=True)

        train_y_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_y.npy'
        test_y_path = '../bin_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_y.npy'

        train_y = np.load(train_y_path, allow_pickle=True)
        test_y = np.load(test_y_path, allow_pickle=True)

        word_vec = np.load('../bin_class/data/wordvec.npy', allow_pickle=True)
        edge_word_vecs = np.load('../bin_class/data/edge_word_vec.npy', allow_pickle=True)
    else:
        train_sentence_ids_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_sentence_ids.npy'
        test_sentence_ids_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_sentence_ids.npy'

        train_sentence_ids = np.load(train_sentence_ids_path, allow_pickle=True)
        test_sentence_ids = np.load(test_sentence_ids_path, allow_pickle=True)

        train_label_ids_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_label_ids.npy'
        test_label_ids_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_label_ids.npy'

        train_label_ids = np.load(train_label_ids_path, allow_pickle=True)
        test_label_ids = np.load(test_label_ids_path, allow_pickle=True)

        train_entity_index_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_entity_index.npy'
        test_entity_index_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_entity_index.npy'

        train_entity_index = np.load(train_entity_index_path, allow_pickle=True)
        test_entity_index = np.load(test_entity_index_path, allow_pickle=True)

        train_y_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'train_y.npy'
        test_y_path = '../five_class' + '/' + 'data/data' + '(' + str(num) + ')' + '/' + 'test_y.npy'

        train_y = np.load(train_y_path, allow_pickle=True)
        test_y = np.load(test_y_path, allow_pickle=True)

        word_vec = np.load('../five_class/data/wordvec.npy', allow_pickle=True)
        edge_word_vecs = np.load('../five_class/data/edge_word_vec.npy', allow_pickle=True)

    if class_num == 2:
        cur_model_path = 'bin_class' + '/' + 'model' + '(' + str(num) + ')'
    else:
        cur_model_path = 'five_class' + '/' + 'model' + '(' + str(num) + ')'
    if not os.path.exists(cur_model_path):
        os.mkdir(cur_model_path)

    # 训练图注意力机制网络
    all_loss = []

    g_rl = tf.Graph()
    sess = tf.Session(graph=g_rl)
    with g_rl.as_default():
        with sess.as_default():

            model = gat_model.Model(word_vec, edge_word_vecs, word_dim, edge_word_dim, max_length,
                                    class_num=class_num, learning_rate=learning_rate)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()

            for epoch in range(Epoch):
                train_loss = []
                time1 = time.time()
                print("training %s epoch..." % str(epoch))
                batch_sentence_ids = []
                batch_label_ids = []
                batch_entity_indexs = []
                batch_y= []
                batch_label_masks = []

                for i, _ in enumerate(tqdm.tqdm(train_sentence_ids)):
                    sentence_ids = train_sentence_ids[i]
                    label_ids = train_label_ids[i]
                    entity_index = train_entity_index[i]
                    y = train_y[i]

                    label_masks = []
                    for m_k in range(len(label_ids)):
                        masks = []
                        for m_t in range(len(label_ids[m_k])):
                            if m_k == m_t:
                                masks.append(1.0)
                            else:
                                if label_ids[m_k][m_t] != 0:
                                    masks.append(1.0)
                                else:
                                    masks.append(0.0)
                        label_masks.append(masks)

                    batch_sentence_ids.append(sentence_ids.tolist())
                    batch_label_ids.append(label_ids.tolist())
                    batch_entity_indexs.append(entity_index.tolist())
                    batch_y.append(y)
                    batch_label_masks.append(label_masks)

                    if (i+1)%batch_size == 0 and i != 0:
                        batch_sentence_ids = np.array(batch_sentence_ids)
                        batch_label_ids = np.array(batch_label_ids)
                        batch_entity_indexs = np.array(batch_entity_indexs)
                        batch_y = np.array(batch_y)
                        batch_label_masks = np.array(batch_label_masks)

                        loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                            model.sentence_ids: batch_sentence_ids,
                            model.label_ids: batch_label_ids,
                            model.entity_index: batch_entity_indexs,
                            model.y: batch_y,
                            model.keep_prob: 0.6,
                            model.label_masks: batch_label_masks,
                        })
                        train_loss.append(loss)

                        batch_sentence_ids = []
                        batch_label_ids = []
                        batch_entity_indexs = []
                        batch_y = []
                        batch_label_masks = []

                losses = np.mean(np.array(train_loss))
                print("train loss: %s" % str(losses))

                all_loss.append(losses)

                time2 = time.time()

                print(time2 - time1)

            true_y = []
            pre_y = []

            if class_num == 2:
                train_result_path = 'bin_class' + '/' + 'model' + '(' + str(num) + ')' + '/' + 'train_result.txt'
            else:
                train_result_path = 'five_class' + '/' + 'model' + '(' + str(num) + ')' + '/' + 'train_result.txt'

            for i, _ in enumerate(tqdm.tqdm(train_sentence_ids[:500])):
                sentence_ids = train_sentence_ids[i]
                label_ids = train_label_ids[i]
                entity_index = train_entity_index[i]
                y = train_y[i]

                label_masks = []
                for m_k in range(len(label_ids)):
                    masks = []
                    for m_t in range(len(label_ids[m_k])):
                        if m_k == m_t:
                            masks.append(1.0)
                        else:
                            if label_ids[m_k][m_t] != 0:
                                masks.append(1.0)
                            else:
                                masks.append(0.0)
                    label_masks.append(masks)
                label_masks = np.array(label_masks)

                sentence_ids = sentence_ids[np.newaxis, :]
                label_ids = label_ids[np.newaxis, :]
                entity_index = entity_index[np.newaxis, :]
                y = y
                label_masks = label_masks[np.newaxis, :]

                pre = sess.run(model.pre, feed_dict={
                    model.sentence_ids: sentence_ids,
                    model.label_ids: label_ids,
                    model.entity_index: entity_index,
                    model.keep_prob: 1.0,
                    model.label_masks: label_masks,
                })

                true_y.append(y)
                pre_y.append(pre[0])

            if class_num == 2:
                accuracy = accuracy_score(true_y, pre_y)
                precision = precision_score(true_y, pre_y, average='binary')
                recall = recall_score(true_y, pre_y, average='binary')
                f1 = f1_score(true_y, pre_y, average='binary')

                print("train set accuracy: %s" % str(accuracy))
                print("train set precision: %s" % str(precision))
                print("train set recall: %s" % str(recall))
                print("train set f1: %s" % str(f1))

                with open(train_result_path, 'a', encoding='utf-8') as f_write:
                    f_write.write('train set')
                    f_write.write('\n')
                    f_write.write('accuracy: ')
                    f_write.write(str(accuracy))
                    f_write.write('\n')
                    f_write.write('precision: ')
                    f_write.write(str(precision))
                    f_write.write('\n')
                    f_write.write('recall: ')
                    f_write.write(str(recall))
                    f_write.write('\n')
                    f_write.write('f1: ')
                    f_write.write(str(f1))
                    f_write.write('\n\n')
                f_write.close()
            elif class_num == 5:
                with open(train_result_path, 'a', encoding='utf-8') as f_write:
                    accuracy = accuracy_score(true_y, pre_y)
                    precision = precision_score(true_y, pre_y, labels=[0,1,2,3,4], average='micro')
                    recall = recall_score(true_y, pre_y, labels=[0,1,2,3,4], average='micro')
                    f1 = f1_score(true_y, pre_y, labels=[0,1,2,3,4], average='micro')

                    print("train set micro accuracy: %s" % str(accuracy))
                    print("train set micro precision: %s" % str(precision))
                    print("train set micro recall: %s" % str(recall))
                    print("train set micro f1: %s" % str(f1))

                    f_write.write('train set')
                    f_write.write('\n')
                    f_write.write('micro:')
                    f_write.write('\n')
                    f_write.write('accuracy: ')
                    f_write.write(str(accuracy))
                    f_write.write('\n')
                    f_write.write('precision: ')
                    f_write.write(str(precision))
                    f_write.write('\n')
                    f_write.write('recall: ')
                    f_write.write(str(recall))
                    f_write.write('\n')
                    f_write.write('f1: ')
                    f_write.write(str(f1))
                    f_write.write('\n\n')

                    accuracy = accuracy_score(true_y, pre_y)
                    precision = precision_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='macro')
                    recall = recall_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='macro')
                    f1 = f1_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='macro')

                    print("train set macro accuracy: %s" % str(accuracy))
                    print("train set macro precision: %s" % str(precision))
                    print("train set macro recall: %s" % str(recall))
                    print("train set macro f1: %s" % str(f1))

                    f_write.write('macro: ')
                    f_write.write('\n')
                    f_write.write('accuracy: ')
                    f_write.write(str(accuracy))
                    f_write.write('\n')
                    f_write.write('precision: ')
                    f_write.write(str(precision))
                    f_write.write('\n')
                    f_write.write('recall: ')
                    f_write.write(str(recall))
                    f_write.write('\n')
                    f_write.write('f1: ')
                    f_write.write(str(f1))
                    f_write.write('\n\n')
                f_write.close()

            true_y = []
            pre_y = []

            for i, _ in enumerate(tqdm.tqdm(test_sentence_ids)):
                sentence_ids = test_sentence_ids[i]
                label_ids = test_label_ids[i]
                entity_index = test_entity_index[i]
                y = test_y[i]

                label_masks = []
                for m_k in range(len(label_ids)):
                    masks = []
                    for m_t in range(len(label_ids[m_k])):
                        if m_k == m_t:
                            masks.append(1.0)
                        else:
                            if label_ids[m_k][m_t] != 0:
                                masks.append(1.0)
                            else:
                                masks.append(0.0)
                    label_masks.append(masks)
                label_masks = np.array(label_masks)

                sentence_ids = sentence_ids[np.newaxis, :]
                label_ids = label_ids[np.newaxis, :]
                entity_index = entity_index[np.newaxis, :]
                y = y
                label_masks = label_masks[np.newaxis, :]

                pre = sess.run(model.pre, feed_dict={
                    model.sentence_ids: sentence_ids,
                    model.label_ids: label_ids,
                    model.entity_index: entity_index,
                    model.keep_prob: 1.0,
                    model.label_masks: label_masks,
                })

                true_y.append(y)
                pre_y.append(pre[0])

            if class_num == 2:
                accuracy = accuracy_score(true_y, pre_y)
                precision = precision_score(true_y, pre_y, average='binary')
                recall = recall_score(true_y, pre_y, average='binary')
                f1 = f1_score(true_y, pre_y, average='binary')

                print("test set accuracy: %s" % str(accuracy))
                print("test set precision: %s" % str(precision))
                print("test set recall: %s" % str(recall))
                print("test set f1: %s" % str(f1))

                with open(train_result_path, 'a', encoding='utf-8') as f_write:
                    f_write.write('test set')
                    f_write.write('\n')
                    f_write.write('accuracy: ')
                    f_write.write(str(accuracy))
                    f_write.write('\n')
                    f_write.write('precision: ')
                    f_write.write(str(precision))
                    f_write.write('\n')
                    f_write.write('recall: ')
                    f_write.write(str(recall))
                    f_write.write('\n')
                    f_write.write('f1: ')
                    f_write.write(str(f1))
                    f_write.write('\n')
                f_write.close()
            elif class_num == 5:
                with open(train_result_path, 'a', encoding='utf-8') as f_write:
                    accuracy = accuracy_score(true_y, pre_y)
                    precision = precision_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='micro')
                    recall = recall_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='micro')
                    f1 = f1_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='micro')

                    print("test set micro accuracy: %s" % str(accuracy))
                    print("test set micro precision: %s" % str(precision))
                    print("test set micro recall: %s" % str(recall))
                    print("test set micro f1: %s" % str(f1))

                    f_write.write('test set')
                    f_write.write('\n')
                    f_write.write('micro:')
                    f_write.write('\n')
                    f_write.write('accuracy: ')
                    f_write.write(str(accuracy))
                    f_write.write('\n')
                    f_write.write('precision: ')
                    f_write.write(str(precision))
                    f_write.write('\n')
                    f_write.write('recall: ')
                    f_write.write(str(recall))
                    f_write.write('\n')
                    f_write.write('f1: ')
                    f_write.write(str(f1))
                    f_write.write('\n\n')

                    accuracy = accuracy_score(true_y, pre_y)
                    precision = precision_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='macro')
                    recall = recall_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='macro')
                    f1 = f1_score(true_y, pre_y, labels=[0, 1, 2, 3, 4], average='macro')

                    print("test set macro accuracy: %s" % str(accuracy))
                    print("test set macro precision: %s" % str(precision))
                    print("test set macro recall: %s" % str(recall))
                    print("test set macro f1: %s" % str(f1))

                    f_write.write('macro:')
                    f_write.write('\n')
                    f_write.write('accuracy: ')
                    f_write.write(str(accuracy))
                    f_write.write('\n')
                    f_write.write('precision: ')
                    f_write.write(str(precision))
                    f_write.write('\n')
                    f_write.write('recall: ')
                    f_write.write(str(recall))
                    f_write.write('\n')
                    f_write.write('f1: ')
                    f_write.write(str(f1))
                    f_write.write('\n')
                f_write.close()

            if class_num == 2:
                save_model_path = 'bin_class' + '/' + 'model' + '(' + str(num) + ')' + '/' + 'model.ckpt'
            else:
                save_model_path = 'five_class' + '/' + 'model' + '(' + str(num) + ')' + '/' + 'model.ckpt'
            saver.save(sess, save_model_path)

    return


def main():
    for i in range(1, 6):
        print("training %s data..."%str(i))
        train(num=i)
    return

if __name__ == '__main__':
    main()