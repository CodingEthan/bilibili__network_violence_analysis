import time
import torch
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self):
        self.train_path = './dataset/train.txt'                                # 训练集
        self.dev_path = './dataset/dev.txt'                                    # 验证集
        self.test_path = './dataset/test.txt'                                  # 测试集
        self.class_list = ['Negative', 'Positive']                                # 类别名单
        # self.save_path = './saved/' + str(int(time.time())) + '.ckpt'        # 模型训练结果
        self.save_path = r'./saved/1669439524.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                          # mini-batch大小128
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.application_path = r'./dataset/application.txt'