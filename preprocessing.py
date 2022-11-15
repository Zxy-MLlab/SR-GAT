import re
import codecs
import json
import numpy as np
import os


class Reader():
    def __init__(self):
        self.data_path_list = ['../../../../origin_data/drug_gene_var/0/data_graph_1', '../../../../origin_data/drug_gene_var/0/data_graph_2',
                          '../../../../origin_data/drug_gene_var/1/data_graph_1', '../../../../origin_data/drug_gene_var/1/data_graph_2',
                          '../../../../origin_data/drug_gene_var/2/data_graph_1', '../../../../origin_data/drug_gene_var/2/data_graph_2',
                          '../../../../origin_data/drug_gene_var/3/data_graph_1', '../../../../origin_data/drug_gene_var/3/data_graph_2',
                          '../../../../origin_data/drug_gene_var/4/data_graph_1', '../../../../origin_data/drug_gene_var/4/data_graph_2']

        self.vec_path = '../../../../vec/vocab_morph.wordvec.st'

        self.class_num = 5
        self.only_single_sent = False
        self.entity_num = 3
        self.word_format = 'lemma'
        self.max_length = 128
        self.edge_vec_dim = 50

        return

    def read_data(self, path):
        all_words = []
        all_lemmas = []
        all_poses = []
        all_in_neigh = []
        all_in_label = []
        all_out_neigh = []  # [batch, node, neigh]
        all_out_label = []  # [batch, node, neigh]
        all_entity_indices = []  # [batch, 3, entity_size]
        all_y = []
        all_nev = 0
        all_pos = 0
        # 创建关系-值键值对
        if self.class_num == 2:
            relation_set = {'resistance or non-response': 0, 'sensitivity': 0, 'response': 0, 'resistance': 0,
                            'None': 1, }
        elif self.class_num == 5:
            relation_set = {'resistance or non-response': 0, 'sensitivity': 1, 'response': 2, 'resistance': 3,
                            'None': 4, }
        else:
            assert False, 'Illegal class num'
        max_words = 0
        max_in_neigh = 0  # 最大入边数量
        max_out_neigh = 0  # 最大出边数量
        max_entity_size = 0
        with codecs.open(path, 'rU', 'utf-8') as f:
            for inst in json.load(f):
                words = []  # 句子中的每个单词
                lemmas = []  # 句子中每个单词的小写
                poses = []  # 句子中每个单词的词性
                if self.only_single_sent and len(inst['sentences']) > 1:
                    continue
                for sentence in inst['sentences']:
                    for node in sentence['nodes']:
                        words.append(node['label'])
                        lemmas.append(node['lemma'])
                        poses.append(node['postag'])
                if len(words) > self.max_length:
                    continue
                max_words = max(max_words, len(words))
                all_words.append(words)
                all_lemmas.append(lemmas)
                all_poses.append(poses)
                in_neigh = [[i, ] for i, _ in enumerate(words)]  # 当前i点入边点集合
                in_label = [['self', ] for i, _ in enumerate(words)]  # 当前i点入边标签集合
                out_neigh = [[i, ] for i, _ in enumerate(words)]  # 当前i点出边点集合
                out_label = [['self', ] for i, _ in enumerate(words)]  # 当前i点出边标签集合
                for sentence in inst['sentences']:
                    for node in sentence['nodes']:
                        i = node['index']
                        for arc in node['arcs']:
                            j = arc['toIndex']
                            l = arc['label']
                            l = l.split('::')[0]
                            l = l.split('_')[0]
                            l = l.split('(')[0]
                            l = l.split(':')[1]
                            if j == -1 or l == '':
                                continue
                            in_neigh[j].append(i)
                            in_label[j].append(l)
                            out_neigh[i].append(j)
                            out_label[i].append(l)
                for _i in in_neigh:
                    max_in_neigh = max(max_in_neigh, len(_i))
                for _o in out_neigh:
                    max_out_neigh = max(max_out_neigh, len(_o))
                all_in_neigh.append(in_neigh)
                all_in_label.append(in_label)
                all_out_neigh.append(out_neigh)
                all_out_label.append(out_label)
                entity_indices = []  # 存储实体最大下标值
                for entity in inst['entities']:
                    entity_indices.append(entity['indices'])
                    max_entity_size = max(max_entity_size, len(entity['indices']))
                assert len(entity_indices) == self.entity_num  # 一个句子中只能有3个实体
                all_entity_indices.append(entity_indices)
                all_y.append(relation_set[inst['relationLabel'].strip()])
                if relation_set[inst['relationLabel'].strip()] == 1:
                    all_pos = all_pos + 1
                else:
                    all_nev = all_nev + 1

        # print("all_nev: %s"%str(all_nev))
        # print("all_pos: %s"%str(all_pos))
        all_lex = all_lemmas if self.word_format == 'lemma' else all_words
        return zip(all_lex, all_poses, all_in_neigh, all_in_label, all_out_neigh, all_out_label, all_entity_indices, all_y), \
            max_words, max_in_neigh, max_out_neigh, max_entity_size

    def read_all_data(self):
        all_instances = []
        max_words = 0
        max_in_neigh = 0
        max_out_neigh = 0
        max_entity_size = 0
        for path in self.data_path_list:
            cur_instances, cur_words_num, cur_in_neigh_num, cur_out_neigh_num, cur_entity_size_num = self.read_data(path)
            all_instances.extend(cur_instances)
            max_words = max(max_words, cur_words_num)
            max_in_neigh = max(max_in_neigh, cur_in_neigh_num)
            max_out_neigh = max(max_out_neigh, cur_out_neigh_num)
            max_entity_size = max(max_entity_size, cur_entity_size_num)
        return all_instances, max_words, max_in_neigh, max_out_neigh, max_entity_size

    def read_vec(self):
        self.word2id = {}  # {"#pad#":0}
        self.id2word = {}  # {0:"#pad#"}

        vec_file = open(self.vec_path, encoding='utf-8')
        word_vecs = {}  # 下标：词向量
        word_vec = []
        for line in vec_file:
            line = line.encode('utf-8').decode('utf-8').strip()
            parts = line.split('\t')
            if len(parts) > 3:
                parts = parts[:2] + parts[-1:]
            cur_index = int(parts[0])
            word = parts[1]
            vector = np.array(list(map(float, re.split('\\s+', parts[2]))), dtype='float32')
            assert word not in self.word2id, word
            self.word2id[word] = cur_index
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
            word_vec.append(vector)
            self.word_dim = vector.size
        vec_file.close()

        word_vec = np.array(word_vec)

        return word_vecs, self.word2id, self.id2word, self.word_dim, word_vec

    def edge_to_words(self, all_instances):
        edge_words = []
        for instance in all_instances:
            in_labels = instance[3]
            out_labels = instance[5]
            for in_label in in_labels:
                edge_words.extend(in_label)
            for out_label in out_labels:
                edge_words.extend(out_label)

        edge_words = list(set(edge_words))

        return edge_words

    def build_edge_vec(self, edge_words):
        assert '#pad#' not in edge_words
        assert 'UNK' not in edge_words

        word2id = {'#pad#': 0, 'UNK': 1, }
        id2word = {0: '#pad#', 1: 'UNK', }

        for word in edge_words:
            cur_index = len(word2id)
            word2id[word] = cur_index
            id2word[cur_index] = word

        vocab_size = len(word2id)
        assert vocab_size == len(edge_words) + 2

        zero_vecs = np.zeros((1, self.edge_vec_dim), dtype=np.float32)
        shape = (vocab_size - 1, self.edge_vec_dim)
        scale = 0.05
        normal_vecs = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
        edge_word_vecs = np.concatenate((zero_vecs, normal_vecs,), axis=0)

        return edge_word_vecs, word2id, id2word, self.edge_vec_dim

class Preprocess():
    def __init__(self, all_instances, word_vecs, word2id, id2word, word_dim, edge_word_vecs, edge_word2id, edge_id2word, edge_word_dim):
        self.all_instances = all_instances

        self.word_vecs = word_vecs
        self.word2id = word2id
        self.id2word = id2word
        self.word_dim = word_dim

        self.edge_word_vecs = edge_word_vecs
        self.edge_word2id = edge_word2id
        self.edge_id2word = edge_id2word
        self.edge_word_dim = edge_word_dim

        self.max_length = 128

        return

    def word_to_vec(self):
        all_sentence_ids = []
        all_label_ids = []
        all_entity_index = []
        all_y = []

        for instance in self.all_instances:
            sentence_ids = []
            label_ids = []
            entity_index = []
            sentence = instance[0]
            in_neigh = instance[2]
            in_label = instance[3]
            entity = instance[6]
            y = instance[7]
            for word in sentence:
                if word not in self.word2id.keys():
                    sentence_ids.append(self.word2id['UNK'])
                else:
                    sentence_ids.append(self.word2id[word.lower()])

            if len(sentence) < self.max_length:
                pad_word_ids = [self.word2id['#pad#'] for x in range(self.max_length - len(sentence))]
                sentence_ids = sentence_ids + pad_word_ids

            for neigh, word_labels in zip(in_neigh, in_label):
                word_label_ids = [self.edge_word2id['#pad#'] for x in range(self.max_length)]

                each_edge_ids = []
                for word in word_labels:
                    if word not in self.edge_word2id.keys():
                        each_edge_ids.append(self.edge_word2id['UNK'])
                    else:
                        each_edge_ids.append(self.edge_word2id[word])

                for i,label_index in enumerate(neigh):
                    word_label_ids[label_index] = each_edge_ids[i]

                label_ids.append(word_label_ids)

            if len(in_label) < self.max_length:
                pad_word_ids = np.zeros((self.max_length-len(in_label), self.max_length), dtype=np.int)
                label_ids = np.array(label_ids)
                label_ids = np.concatenate((label_ids, pad_word_ids), axis=0)
                label_ids = label_ids.tolist()

            for each_entity in entity:
                for index in each_entity:
                    entity_index.append(sentence_ids[index])

            if len(entity_index) < self.max_length:
                pad_word_ids = [0 for x in range(self.max_length - len(entity_index))]
                entity_index = entity_index + pad_word_ids


            all_sentence_ids.append(sentence_ids)  #[num, 128]
            all_label_ids.append(label_ids)     #[num, 128, 128]
            all_entity_index.append(entity_index)  #[num, 128]
            all_y.append(y)  #[num, ]

        return all_sentence_ids, all_label_ids, all_entity_index, all_y

    def create_train_test_data(self, all_sentence_ids, all_label_ids, all_entity_index, all_y):
        state = np.random.get_state()
        np.random.shuffle(all_sentence_ids)

        np.random.set_state(state)
        np.random.shuffle(all_label_ids)

        np.random.set_state(state)
        np.random.shuffle(all_entity_index)

        np.random.set_state(state)
        np.random.shuffle(all_y)

        data_size = len(all_sentence_ids)
        part_size = int(data_size / 5)

        print("now save data ...")

        for i in range(5):
            test_sentence_ids = all_sentence_ids[i*part_size : (i+1)*part_size]
            train_sentence_ids = all_sentence_ids[: i*part_size]
            train_sentence_ids = train_sentence_ids + all_sentence_ids[(i+1)*part_size: ]

            test_label_ids = all_label_ids[i*part_size : (i+1)*part_size]
            train_label_ids = all_label_ids[: i*part_size]
            train_label_ids = train_label_ids + all_label_ids[(i+1)*part_size: ]

            test_entity_index = all_entity_index[i*part_size : (i+1)*part_size]
            train_entity_index = all_entity_index[: i*part_size]
            train_entity_index = train_entity_index + all_entity_index[(i+1)*part_size: ]

            test_y = all_y[i*part_size : (i+1)*part_size]
            train_y = all_y[: i*part_size]
            train_y = train_y + all_y[(i+1)*part_size: ]

            train_sentence_ids = np.array(train_sentence_ids)
            test_sentence_ids = np.array(test_sentence_ids)

            train_label_ids = np.array(train_label_ids)
            test_label_ids = np.array(test_label_ids)

            train_entity_index = np.array(train_entity_index)
            test_entity_index = np.array(test_entity_index)

            train_y = np.array(train_y)
            test_y = np.array(test_y)

            data_save_path = 'data' + '/' + 'data' + '(' + str(i+1) + ')'
            os.mkdir(data_save_path)

            train_sentence_ids_path = data_save_path + '/' + 'train_sentence_ids.npy'
            test_sentence_ids_path = data_save_path + '/' + 'test_sentence_ids.npy'

            train_label_ids_path = data_save_path + '/' + 'train_label_ids.npy'
            test_label_ids_path = data_save_path + '/' + 'test_label_ids.npy'

            train_entity_index_path = data_save_path + '/' + 'train_entity_index.npy'
            test_entity_index_path = data_save_path + '/' + 'test_entity_index.npy'

            train_y_path = data_save_path + '/' + 'train_y.npy'
            test_y_path = data_save_path + '/' + 'test_y.npy'

            np.save(train_sentence_ids_path, train_sentence_ids)
            np.save(test_sentence_ids_path, test_sentence_ids)

            np.save(train_label_ids_path, train_label_ids)
            np.save(test_label_ids_path, test_label_ids)

            np.save(train_entity_index_path, train_entity_index)
            np.save(test_entity_index_path, test_entity_index)

            np.save(train_y_path, train_y)
            np.save(test_y_path, test_y)

        return

    def save_ebd(self, word_vec, edge_word_vecs, word2id, id2word, edge_word2id, edge_id2word):
        np.save('data/wordvec.npy', word_vec)
        np.save('data/edge_word_vec.npy', edge_word_vecs)

        with open('data/word2ids.json', 'a', encoding='utf-8') as f_write:
            json.dump(word2id, f_write)
        f_write.close()

        with open('data/id2words.json', 'a', encoding='utf-8') as f_write:
            json.dump(id2word, f_write)
        f_write.close()

        with open('data/edge_word2ids.json', 'a', encoding='utf-8') as f_write:
            json.dump(edge_word2id, f_write)
        f_write.close()

        with open('data/edge_id2words.json', 'a', encoding='utf-8') as f_write:
            json.dump(edge_id2word, f_write)
        f_write.close()

        return

reader = Reader()
all_instances, _, _, _, _ = reader.read_all_data()

edge_words = reader.edge_to_words(all_instances)
word_vecs, word2id, id2word, word_dim, word_vec = reader.read_vec()
edge_word_vecs, edge_word2id, edge_id2word, edge_word_dim = reader.build_edge_vec(edge_words)
preprocess = Preprocess(all_instances, word_vecs, word2id, id2word, word_dim, edge_word_vecs, edge_word2id,
                            edge_id2word, edge_word_dim)
all_sentence_ids, all_label_ids, all_entity_index, all_y = preprocess.word_to_vec()

preprocess.create_train_test_data(
        all_sentence_ids, all_label_ids, all_entity_index, all_y)

preprocess.save_ebd(word_vec, edge_word_vecs, word2id, id2word, edge_word2id, edge_id2word)