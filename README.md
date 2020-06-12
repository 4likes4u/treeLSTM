# treeLSTM
Personal implementation of the ACL 19 paper "Tree LSTMs with Convolution Units to Predict Stance and Rumor Veracity in Social Media Conversations" with pytorch.

# Code Structure
```
root
├── process_data
│   ├── generate_tree.pys
│   ├── get_label.py
│   └── SKPencoder4tree.py
├── treeLSTM.py
└── readme.md
```
## generate_tree.py
Convert raw data into 1) raw tweets (output: "event.tweets") 2) clean row tree (output:  "event_tree.json")  3) clean original tree (output: "event_ori_tree.json")

*p.s. there are some threads contains more than one main tweets (which means it's not a tree structure anymore), I only keep one tweet and generate the tree.*
## get_label.py
Get the label for the whole data in tree (json) form (output: "event_label.txt").

"p.s. there are some threads lack of the label in annotation.json, I label them with class 4 to keep the tree structure. Class 4 will be ingored during training and validating."
## SKPencoder4tree.py
Convert the raw tweets into SKP feature by tf-SKP in following repo [https://github.com/tensorflow/models/tree/master/research/skip_thoughts], output: "events.npy"

*The authors in ACL used theano based version, however theano is not suitable for python3.x and GPU, so I just use the tf version*
## treeLSTM.py
I implemented two models: Child Conv Tree LSTM and Child Sum Tree LSTM, performances will be print on the screen.

*It will be hard to run the treeLSTM parallelly in GPU, so I keep batch size equal to 1 and use CPU to run this model. Due to the size of training sample, it won't cost a lot to train the model (less than one hour for 30 epochs).*

# Performance
