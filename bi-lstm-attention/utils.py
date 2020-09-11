import re


def get_data(f):
  sents, labels =[], []
  for line in f.readlines():
    line = re.sub('\n', '', line)
    _, sent, label = line.split('\t')
    sents.append(sent)
    labels.append(label)
  return sents, labels

def make_vocab(train_sentences, min_count = 1):
  vocab_set = set()
  vocab_num = dict()

  for sent in train_sentences:
    for token in sent.split(' '):
      if token not in vocab_num:
        vocab_num[token] = 1
      else:
        ori_num = vocab_num[token]
        ori_num += 1
        vocab_num[token] = ori_num        

    for vocab in vocab_num:
      if vocab_num[vocab] > min_count:
        vocab_set.add(vocab)

  fw = open('./data/vocab.txt', 'w', encoding='utf8')
  for vocab in vocab_set:
    fw.write(vocab + '\n')
  fw.close()

  return vocab_set

