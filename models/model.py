import torch
import torch.nn as nn
from transformers import BertModel

class TextClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(TextClassifier, self).__init__()
        model_path = 'E:\Chinese_word2vec\JointMatch-main\pre\pretrained_bert'
        self.bert = BertModel.from_pretrained(model_path)
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        predict = self.linear(pooled_output)
        return predict