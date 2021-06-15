from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids,
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True,
                            attention_mask=attention_mask
                            )
        # outputs:
        # attentions (bs, num_heads=12, seq_len, seq_len) * tuple(12)
        # hidden_states (bs, seq_len, hidden_state) * tuple(13)
        # last_hidden_state (bs, seq_len, hidden_state)
        # pooler_output (bs, hidden_state)
        return outputs


class cls(nn.Module):
    def __init__(self):
        super(cls, self).__init__()

        self.state = nn.Linear(768, 48)
        self.dropout = nn.Dropout(0.2)

        self.critic = nn.Linear(48, 5)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        x = self.state(hidden_states)  # (bs, seqlen, 48)
        x = F.relu(x)

        logits = self.critic(x)  # (bs, seqlen, 1)

        return logits
