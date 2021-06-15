import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import torch

def result_type(y_true, y_pred, save_right=False):
    # tn: predict causal label to 'O' label
    # fp: predict 'O' label to causal label

    for y_t, y_p in zip(y_true, y_pred):
        if y_t in [1, 2, 3, 4] and y_p == 0:
            return 'TN'
        elif y_t == 0 and y_p in [1, 2, 3, 4]:
            return 'FP'

    # save right prediction
    if save_right:
        for y_t in y_true:
            if y_t in [1, 2, 3, 4]:
                return 'TRUE'
    return ''

def process_ids(sent, seqlen, tokenizer, attention):
    # sent of ids -> sent of str, sent need to be truncate to len
    sent = sent[:seqlen]
    sent = tokenizer.convert_ids_to_tokens(sent)

    # attention (heads=12, seqlen, seqlen) maxpool, truncate
    attention = torch.max(attention, dim=0)[0]  # (seqlen, seqlen)
    attention = attention[:seqlen, :seqlen]
    attention.softmax(dim=1)
    attention = attention.cpu().numpy()

    return sent, attention

def view_attention(sent, attention, save_name):

    df = pd.DataFrame(attention, columns=sent, index=sent)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    # cax = ax.matshow(df)
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # fontdict = {'rotation': 'vertical'}    #设置文字旋转
    fontdict = {'rotation': 90}  # 或者这样设置文字旋转
    # ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
    # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
    ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
    ax.set_yticklabels([''] + list(df.index))

    plt.savefig(f'/data/zhaoy/myproject/stratified_bert/result_analysis/{save_name}.png')
    plt.close()

def save_result(fp, res_type, sub_y, sub_y_h, sent, att, wrong_num,
                save_picture=False, input_ids=None, rl_mask=False):
    # input ids must be used with rl mask together

    if res_type == 'TN' or res_type == 'FP' or res_type == 'TRUE':
        # find none zero index
        a = (sub_y >= 1).nonzero().view(-1)
        for index in a:
            sent[index] = sent[index] + '_t'
        b = (sub_y_h >= 1).nonzero().view(-1)
        for index in b:
            sent[index] = sent[index] + '_p'

        fp.write(' '.join(sent) + '\n')
        # add rl mask label
        if rl_mask:
            mask_sent = []
            for index, word in enumerate(sent):
                if input_ids[index] == 103:
                    mask_sent.append('[MASK]')
                else:
                    mask_sent.append(word)
            fp.write(' '.join(mask_sent) + '\n')

        fp.write('y_true: ' + str(sub_y.cpu().tolist()) + '\n')
        fp.write('y_pred: ' + str(sub_y_h.cpu().tolist()) + '\n')


        # save attention weight picture
        if save_picture and (res_type == 'TRUE'):
            view_attention(sent, att, str(wrong_num) + res_type)

        if res_type != 'TRUE':
            return wrong_num + 1

    return wrong_num