import json
import torch
import numpy as np
from torch import nn
import torch.utils.data as Data
from torch.nn import init
from sklearn.metrics import f1_score
import random
"""
use json as the tree structure

"""
def eval_at_dev(model, val_feature, val_whole_tree, gd_label_arr):
    """
	args: val_feature numpy array: (thtread_size, tweets_size, 4800-d features)
	gd_label_arr: [label for thread_1, label for thread_2]
    """
    pred_list = []
    model.eval()
    for val_index in range(val_feature.shape[0]):
        pred = model.get_hidden_buffer(val_feature[val_index], val_whole_tree[val_index]['0'], 0)#get model predict
        pred_index = torch.topk(pred[:, 0:4], 1)[1].view(-1,).tolist()
        pred_list = pred_list +  pred_index
    f1 = f1_score(gd_label_arr, pred_list, labels=[0,1,2,3], average='macro')
    print("current model over dev set macro f measure: " + str(f1))
    return f1

class TreeLSTMCell(nn.Module): 
    def __init__(self):
        super(LSTM, self).__init__()
    def __init__(self, input_size = 4800, hidden_size = 64, class_num = 5):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        # bais term only at W is ok, U doesn't need lah
        self.W_i = torch.nn.Linear(self.input_size,  self.hidden_size)
        self.U_i = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_f = torch.nn.Linear(self.input_size, self.hidden_size)
        self.U_f = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = torch.nn.Linear(self.input_size,  self.hidden_size)
        self.U_o = torch.nn.Linear(self.hidden_size,  self.hidden_size, bias=False)
        self.W_u = torch.nn.Linear(self.input_size,  self.hidden_size)
        self.U_u = torch.nn.Linear(self.hidden_size,  self.hidden_size, bias=False)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.hidden_size, kernel_size=(2, self.hidden_size))
        self.hidden_buffer = []
        # self.fc_1 = torch.nn.Linear(self.hidden_size, 128)
        self.classifier = torch.nn.Linear(self.hidden_size,  self.class_num)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.sum = False
    def forward(self, inputs, tree, current_child_id):
        # work for batch_size = 1
        # compute by recursive, to compute current node, we must have the child, i.e.: tree[child_id]
        inputs = torch.Tensor(inputs) # change array from nparray to torch.Tensor
        batch_size = 1
        children_outputs = [self.forward(inputs, tree[child_id], child_id)
                            for child_id in tree] # shape: child_num*
        # if currently we are at non-leaf nodes, then we got children_states,
        # or we should initalize it with torch.Tensor
        if children_outputs:
            children_states = children_outputs
        else:
            children_states = [(torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))]

        #given the children states, how we compute the hidden states
        return self.node_forward(inputs[int(current_child_id), :], children_states)

    def node_forward(self, inputs, children_states):
        # comment notation:
        # inputs: 4800d vector
        # children_states: K*[(C-dim hidden states, C-dim cell memory) ]
        # C for hidden state dimensions
        # K for number of children
        # calculate gate outputs for i, o, u
        batch_size = 1
        # Child Sum LSTM
        if(self.sum):
            K = len(children_states)
            average_h = torch.zeros(batch_size, self.hidden_size)
            K = len(children_states)
            # average to get
            for index in range(int(K)):
                average_h = average_h + children_states[index][0]
        # Child Conv LSTM
        else:
        # Child Conv LSTM
            K = len(children_states)
            if(K < 2): # if only one child, cannot conv, just return
                child_tensor_list = []
                child_tensor_list.append(children_states[0][0])
                child_tensor_list.append(children_states[0][0])
                child_tensor = torch.stack(child_tensor_list).view(1, 1, 2, self.hidden_size)
            else:
                child_tensor_list = []
                #get the matrix
                for index in range(int(K)):
                    child_tensor_list.append(children_states[index][0])
                child_tensor = torch.stack(child_tensor_list).view(1, 1, K, self.hidden_size) #batch x channel x children_size x hidden_size
            # get the conv output
            child_tensor_conv = self.conv(child_tensor) # batch x channel x children_size x hidden_size
            # pooling by max
            average_h = torch.max(child_tensor_conv, dim=2)[0] # batch_size x hidden_size
            average_h = average_h.view(1, -1)
        # compute each gate
        i = torch.sigmoid(self.W_i(inputs) + self.U_i(average_h))
        o = torch.sigmoid(self.W_o(inputs) + self.U_o(average_h))
        u = torch.tanh(self.W_u(inputs) + self.U_u(average_h))

        # forget gate needs different computing, for each children differently
        sum_f = torch.zeros(batch_size, self.hidden_size)
        for index in range(int(K)):
            f = torch.sigmoid(self.W_f(inputs) + self.U_f(children_states[index][0]))
            sum_f = sum_f + f*children_states[index][1]

        # calculate cell state and hidden state
        c = sum_f + i*u
        h = o*torch.tanh(c)
        cell_memory = c
        hidden_state = h
        output = hidden_state#torch.relu(self.fc_1(hidden_state))
        output = self.dropout(output)
        output = self.classifier(output)
        self.hidden_buffer.append(output.view(-1,))
        return (hidden_state, cell_memory)

    def get_hidden_buffer(self, inputs, tree, current_child_id):
        self.hidden_buffer = [] # empty it
        self.forward(inputs, tree, current_child_id)
        return torch.stack(self.hidden_buffer)

cd = TreeLSTMCell()
#init.orthogonal_(cd.parameters)

#loss = nn.CrossEntropyLoss(ignore_index = 4)
loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1, 0.5, 0]), ignore_index=4)

#定义优化算法
import torch.optim as optim #
optimizer = optim.Adam(cd.parameters(), lr=0.001) #使用Adam 优化器

# read data
#define train and val events
five_events = ['ch', 'fg', 'gc', 'ow', 'ss']
dev_events = 0 # means for ch
train_events = [1, 2, 3, 4]

#read in dev set

dev_feature = np.load(five_events[dev_events] + ".npy")
dev_tree_in =  open(five_events[dev_events] + '_tree.json', 'r')
dev_whole_tree = []
for line in dev_tree_in:
    line = line.strip()
    dev_whole_tree.append(json.loads(line))
dev_label_set = []
dev_label_in = open(five_events[dev_events] + '_label.txt', 'r')
dev_label_arr = []
for line in dev_label_in:
    dev_label_set.append([int(x) for x in line.strip().split("\t")])
    dev_label_arr = dev_label_arr + [int(x) for x in line.strip().split("\t")]


#read in train set

train_whole_tree = []
train_label_set = []
train_label_arr = []

for eve_index in range(0, 4):

    train_tree_in =  open(five_events[train_events[eve_index]] + '_tree.json', 'r')
    train_label_in = open(five_events[train_events[eve_index]] + '_label.txt', 'r')

    for line in train_tree_in:
        line = line.strip()
        train_whole_tree.append(json.loads(line))

    for line in train_label_in:
        train_label_set.append([int(x) for x in line.strip().split("\t")])
        train_label_arr = train_label_arr + [int(x) for x in line.strip().split("\t")]
    train_tree_in.close()
    train_label_in.close()

# v2 is the original data after tweets padding
train_feature = np.concatenate((np.load(five_events[train_events[0]] + "v2.npy"), 
    np.load(five_events[train_events[1]] + "v2.npy"),
    np.load(five_events[train_events[2]] + "v2.npy"),
    np.load(five_events[train_events[3]] + "v2.npy")))

# train
max_epoch = 30
train_index_set = [i for i in range(0, train_feature.shape[0])]
random.shuffle(train_index_set)
for epoch in range(max_epoch):
    
    for index in train_index_set:
        output = cd.get_hidden_buffer(train_feature[index], train_whole_tree[index]['0'], 0)
        L = loss(output, torch.Tensor(train_label_set[index]).long())
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
    eval_at_dev(cd, dev_feature, dev_whole_tree, dev_label_arr)
    print('epoch: %d, loss: %f' % (epoch, L.item()))
