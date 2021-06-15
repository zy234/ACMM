import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
sys.path.append(r'/data/zhaoy/myproject/ACMM/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import argparse

from bert_encoder import BertEncoder, cls
from utils import CeDataSet, pad, eval_matrics
from view_result import result_type, process_ids, save_result
from a2c_model import Envirment, A2CModel, RLtrainIters_v2, RL_optimize, ReplayMemory

def rl_train_step(epoch, **inputs):
    print(r'Start RL train step:')
    inputs['a2c'].train()
    inputs['env'].eval()

    ave_loss = 0.
    total_reward = 0.
    total_step = 0
    mask_nums = 0
    memory = ReplayMemory(60)
    for bs, batch in enumerate(inputs['train_iter']):

        word, x, label, y, attention_mask, seq_lens, y_sent = batch

        for i in range(len(x)):
            RLtrainIters_v2(inputs['env'], x[i], inputs['a2c'], attention_mask[i], y[i], seq_lens[i], memory)

        flag = False
        reward = 0
        mask_num = 0
        if bs % 48 == 0:
            flag = True
            reward, mask_num = RL_optimize(inputs['a2c'], inputs['env'], memory, inputs['optimizer'], pri=flag)
        total_reward += reward
        mask_nums += mask_num
        total_step += 1


    print(f'epoch {epoch} is end, average loss is {ave_loss / total_step}, total reward is {total_reward}')
    print(f'mask number is {mask_nums}')

    # return total_reward

def train_step(epoch, only_encoder=False, **inputs):
    print(r'Start model train step:')
    inputs['env'].train()
    inputs['a2c'].eval()

    ave_loss = 0.
    total_step = 0
    total_mask_num = 0

    for _, batch in enumerate(inputs['train_iter']):

        word, x, label, y, attention_mask, seq_lens, y_sent = batch
        seed = torch.rand(1)
        if only_encoder:
            seed = 0

        # get new rl mask ids, attention mask
        if seed > 0.5:
            with torch.no_grad():
                x, mask_num = inputs['a2c'].get_new_ids(x, seq_lens, inputs['a2c'])
            total_mask_num += mask_num

            loss, _ = inputs['env'](x, attention_mask, y)

            loss = loss.mean(dim=-1)
        # normal training without mask
        else:
            # inputs['env'].train()
            loss, _ = inputs['env'](x, attention_mask, y)
            loss = loss.mean(dim=-1)

        inputs['optimizer'].zero_grad()
        loss.backward()
        inputs['optimizer'].step()

        ave_loss += loss.item()
        total_step += 1

    print(f'epoch {epoch} is end, average loss is {ave_loss / total_step}')
    print(f'mask num is {total_mask_num}')
def eval_step(**inputs):
    inputs['env'].eval()
    inputs['a2c'].eval()

    Y_true, Y_preds = [], []
    with torch.no_grad():
        for i, batch in enumerate(inputs['eval_iter']):
            word, x, label, y, attention_mask, seq_lens, y_sent = batch

            logits= inputs['env'](x, attention_mask, y, predict=True)
            y_hat = torch.argmax(logits, -1)  # (bs, seq_len)

            for y_h, y_t in zip(y_hat, y):
                y_true = [inputs['label_list'][i] for i in y_t if i != -100]
                y_pred = [inputs['label_list'][j] for i, j in zip(y_t, y_h) if i != -100]

                Y_true.append(y_true)
                Y_preds.append(y_pred)

    precision, recall, f1 = eval_matrics(Y_true, Y_preds)

    print(f' precision: {precision}\n recall: {recall} \n f1 score: {f1}')

    return precision, recall, f1


def train(encoder, cls, optimizer, train_iter, eval_iter, hp, a2c):

    patience_num = 0
    best_f1 = 0
    env = Envirment(encoder, cls)

    rl_optimizer = optim.Adam(a2c.parameters(), lr=0.03)

    # warm up of encoder
    for epoch in range(hp.encoder_warm_up):
        print('=' * 40)
        print('warm-up encoder...')
        train_step(epoch,
                   only_encoder=True,
                   env=env,
                   a2c=a2c,
                   optimizer=optimizer,
                   train_iter=train_iter,
                   device=hp.device
                   )
    precision, recall, f1 = eval_step(env=env,
                                      a2c=a2c,
                                      eval_iter=eval_iter,
                                      label_list=hp.label_list)
    print(f'encoder warm-up is done, f1 score is {f1}.')

    # warm up of mask model
    for epoch in  range(hp.mask_warm_up):
        print('=' * 40)
        print('warm-up mask model...')
        rl_train_step(epoch,
                      env=env,
                      a2c=a2c,
                      optimizer=rl_optimizer,
                      train_iter=train_iter,
                      device=hp.device)
    print(f'mask model warm-up is done')

    # adversarial train step
    for epoch in range(hp.epochs):
        print(f'====================epoch{epoch}====================')
        rl_train_step(epoch,
                      env=env,
                      a2c=a2c,
                      optimizer=rl_optimizer,
                      train_iter=train_iter,
                      device=hp.device)

        train_step(epoch,
                   env=env,
                   a2c=a2c,
                   optimizer=optimizer,
                   train_iter=train_iter,
                   device=hp.device)

        precision, recall, f1 = eval_step(env=env,
                                          a2c=a2c,
                                          eval_iter=eval_iter,
                                          label_list=hp.label_list)

        if best_f1 < f1:
            best_f1 = f1
            patience_num = 0
            print('model update!!!')
            # delete previous model, the name start with 'model'
            for file in os.listdir('./result'):
                if file.startswith('model'):
                    os.remove(os.path.join('./result', file))
            # save model
            torch.save({'cls_state_dict': cls.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'a2c_state_dict': a2c.state_dict()},
                       f'./result/model-f1-%.5f.pt'%best_f1)

        else:
            patience_num += 1

        if patience_num > hp.patience:
            print(f'training is end, program is early stop at epoch {epoch}, best f1 is {best_f1}')
            break

    # final file rename
    for file in os.listdir('./result'):
        if file.startswith('model'):
            os.rename(os.path.join('./result', file),
                      os.path.join('./result', f'a2c-' + file))

def eval(encoder, a2c_model, a2c, eval_iter, hp, tokenizer):
    '''
    :param encoder:
    :param a2c_model: cls
    :param a2c:
    :param eval_iter:
    :param hp:
    :param tokenizer:
    :return:
    '''
    encoder.eval()
    a2c_model.eval()
    a2c.eval()

    Y_true, Y_preds, Y_sent_true, Y_sent_preds = [], [], [], []
    fp = open('./result_analysis/result.txt', 'w')
    wrong_num = 0
    total_mask_num = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_iter):
            # if i == 20: break
            word, x, label, y, attention_mask, seq_lens, y_sent = batch

            out_puts = encoder(x, attention_mask)

            input_ids, mask_num = a2c.get_new_ids(x, seq_lens, a2c)
            total_mask_num += mask_num

            # part ce model
            logits = a2c_model(out_puts['last_hidden_state'])
            # logits = ce_out['logits']  # (bs, seq_len, 5)
            y_hat = torch.argmax(logits, -1)  # (bs, seq_len)

            # get result analysis
            for sub_x, seq_len, sub_att, sub_y, sub_y_h, sub_mask_ids in zip(
                    x, seq_lens, out_puts['attentions'][11], y, y_hat, input_ids):

                sub_y = sub_y[:seq_len]
                sub_y_h = sub_y_h[:seq_len]

                res_type = result_type(sub_y, sub_y_h, save_right=True)
                sent, att = process_ids(sub_x, seq_len, tokenizer, sub_att)
                wrong_num = save_result(fp, res_type, sub_y, sub_y_h, sent, att, wrong_num,
                                        input_ids=sub_mask_ids, rl_mask=True, save_picture=False)

            # get all label list
            for y_h, y_t in zip(y_hat, y):
                y_true = [hp.label_list[i] for i in y_t if i != -100]
                y_pred = [hp.label_list[j] for i, j in zip(y_t, y_h) if i != -100]

                Y_true.append(y_true)
                Y_preds.append(y_pred)

    fp.close()
    precision, recall, f1 = eval_matrics(Y_true, Y_preds)

    print(f' precision: {precision}\n recall: {recall} \n f1 score: {f1}, wrong label num {wrong_num}')
    print(f'mask number is {total_mask_num}')
    print('=' * 20)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--label_list', default=['O', 'B-Cause', 'I-Cause', 'B-Effect', 'I-Effect'])
    parser.add_argument('--mask_warm_up', type=int, default=10)
    parser.add_argument('--encoder_warm_up', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=130)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--eval_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    hp = parser.parse_args()

    hp.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get train test data iter
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CeDataSet(os.path.join(hp.data_path, 'train.txt'), tokenizer,
                              hp.label_list)
    test_dataset = CeDataSet(os.path.join(hp.data_path, 'test.txt'), tokenizer,
                              hp.label_list)
    train_iter = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad)
    eval_iter = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=pad)

    # for i in range(13):
    encoder = BertEncoder().to(hp.device)
    cls = cls().to(hp.device)
    optimizer = optim.Adam([{'params': encoder.parameters()},
                            {'params': cls.parameters()}], lr=1e-5)
    a2c = A2CModel().to(hp.device)
    # eval func
    if hp.eval:
        model_dict = torch.load(hp.eval_path)
        encoder.load_state_dict(model_dict['encoder_state_dict'])
        cls.load_state_dict(model_dict['cls_state_dict'])
        eval(encoder, cls, a2c, eval_iter, hp, tokenizer)
    # train func
    else:
        train(encoder, cls, optimizer, train_iter, eval_iter, hp, a2c)