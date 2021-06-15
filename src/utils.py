from torch.utils.data import Dataset
from transformers import BertTokenizer
import sys
import torch
from seqeval.metrics import precision_score as word_p, recall_score as word_r, f1_score as word_f1
from sklearn.metrics import precision_score as sent_p, recall_score as sent_r, f1_score as sent_f1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CeDataSet(Dataset):
    def __init__(self, data_path, tokenizer, labels, max_len=256):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2ids = {label: ids for ids, label in enumerate(labels)}

        with open(data_path, 'r', encoding='utf-8', errors='ignore') as fr:
            entries = fr.read().strip().split('\n\n')

        sents, tags = [], []
        for entrie in entries:
            # has punctuation, not delete yet
            words = [line.split()[0] for line in entrie.splitlines()]
            tag = [line.split()[1] for line in entrie.splitlines()]

            sents.append(words[:self.max_len])
            tags.append(tag[:self.max_len])

        self.sents = sents
        self.tags = tags

    def __getitem__(self, idx):
        sent, tag = self.sents[idx], self.tags[idx]  # sigle sentence, ['he', 'is', 'a', 'man']
        x, y = [], []
        # is_heads = []  # first token of tokens is 1, others 0

        for w, t in zip(sent, tag):
            tokens = self.tokenizer.tokenize(w)  # change word to tokens
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # change tokens to ids, may more than 1
            # is_head = [1] + [0] * (len(tokens) - 1)
            tag_ids = self.label2ids[t]  # convert label to ids
            tag_ids = [tag_ids] + [-100] * (len(tokens) - 1)

            x.extend(tokens_ids)
            y.extend(tag_ids)
            # is_heads.extend(is_head)

        x = self.tokenizer.build_inputs_with_special_tokens(x)
        y = self.add_special_token_for_label(y)
        attention_mask = [1 for i in x]
        assert len(x) == len(y), f'len of x is {len(x)}, len of y is {len(y)}'

        words = ' '.join(sent)
        labels = ' '.join(tag)
        seq_len = len(x)
        return words, x, labels, y, attention_mask, seq_len

    def __len__(self):
        return len(self.sents)

    def add_special_token_for_label(self, labels):
        # list of int
        labels.insert(0, -100)
        labels.append(-100)

        return labels

    def data_discription(self):
        # data details
        data_len = len(self.tags)
        causal_len, other_len = 0, 0
        for tag in self.tags:
            if 'B-Cause' in tag:
                causal_len += 1
            else:
                other_len += 1

        print('=' * 40)
        print(f"There are a total of {data_len} sentences, "
              f"of which {causal_len} has a causal relationship and {other_len} has no causal relationship")


def pad(batch):
    # get all items
    f = lambda x: [sample[x] for sample in batch]
    word, x, labels, y, attention_mask, seq_lens = f(0), f(1), f(2), f(3), f(4), f(5)
    y_sent = []
    # pad to max_len
    max_len = max(seq_lens)
    for sub_x in x:
        sub_x.extend([0] * (max_len - len(sub_x)))
    for sub_att in attention_mask:
        sub_att.extend([0] * (max_len - len(sub_att)))
    for sub_y in y:
        # prepare labels for sentence classification
        if 1 in sub_y or 3 in sub_y:
            y_sent.append(1)
        else:
            y_sent.append(0)
        sub_y.extend([-100] * (max_len - len(sub_y)))
    assert len(y_sent) == len(seq_lens)

    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    y_sent = torch.tensor(y_sent).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)

    return word, x, labels, y, attention_mask, seq_lens, y_sent

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# data = CeDataSet('../data/semeval-2010/test.txt', tokenizer,
#                  labels=['O', 'B-Cause', 'I-Cause', 'B-Effect', 'I-Effect'])
# data.__getitem__(31)

def eval_matrics(y_true, y_pred, type='ce'):
    if type == 'ce':
        return (word_p(y_true, y_pred, average='macro'), word_r(y_true, y_pred, average='macro'), word_f1(y_true, y_pred, average='macro'))
    elif type == 'cs':
        return (sent_p(y_true, y_pred), sent_r(y_true, y_pred), sent_f1(y_true, y_pred))

    print(f"There is no type as {type}!")
    sys.exit()