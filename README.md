# treeLSTM
Personal implementation of the ACL 19 paper "Tree LSTMs with Convolution Units to Predict Stance and Rumor Veracity in Social Media Conversations" with pytorch.

# Performance
|Model|CH|SS|FG|OS|GC|
| ---------- | :----------: | :-----------:  | :-----------: | :-----------:  | :-----------: |
|Our Implementation   |  0.511   |   0.470 |  0.471  |  0.484  |  0.559  |
|Original Paper report|  0.514   |   0.579 |  0.553  |  0.469  |  0.547  |


The performance drops a lot compare to the orignal reported one, may due to following possible reasons:

1. Possible bugs I am not find currently in my implementation. : )
2. (1) Different deep learning frameworks: They use DGL and pytorch to build the treelstm, however I build the whole model by pytorch purely. (2) Different tweets encoding methods: I adopt SKP (the best encoding method in the Table 3, however, the version of my SKP based on tensorflow but the authors' is theano.
3. Confused details in the paper: The author didn't report the batch size to train the model, and the initialization method of the first      hidden state and cell memory (i.e., h_{0}, c_{0}). 
   
   The most important one part - resampling the minor class is also missing, this part is very confused as described in paper section 3, so I just turn the weight of class comment as 0.5 and 1 for other classes, in my experiments this rate affects a lot. 
   
   And in the Tree Conv LSTM, how to do the convolution operation if there only exits one children, the kernels size is 2！
4. Different training and development set: Due to the data is not clean in the raw form, some threads contains more than one source tweets, I just keep the first main tweets, this may cause the dataset not same as the authors use. 
   
   Additionally, I have to argue that the dataset in baseline Lukasik et al., 2016 is different from the author used (sample size in each class cannot match). So personally I think the compare between two method is meaningless. 
5. *Just a question: in my view use 4 events to train and evaluate at another event is ill defined, because it will lead the model tuning hyper parameters at testing set.*


# Code Structure
```
root
├── process_data
│   ├── generate_tree.pys
│   ├── get_label.py
│   └── SKPencoder4tree.py
├── treeLSTM.py
├── log # training logs
└── readme.md
```
### generate_tree.py
Convert raw data into 1) raw tweets (output: "event.tweets") 2) clean row tree (output:  "event_tree.json")  3) clean original tree (output: "event_ori_tree.json")

*p.s. there are some threads contains more than one main tweets (which means it's not a tree structure anymore), I only keep one tweet and generate the tree.*
### get_label.py
Get the label for the whole data in tree (json) form (output: "event_label.txt").

*p.s. there are some threads lack of the label in annotation.json, I label them with class 4 to keep the tree structure. Class 4 will be ingored during training and validating.*
### SKPencoder4tree.py
Convert the raw tweets into SKP feature by tf-SKP in following repo [https://github.com/tensorflow/models/tree/master/research/skip_thoughts], output: "events.npy"

*The authors in ACL used theano based version, however theano is not suitable for python3.x and GPU, so I just use the tf version*
### treeLSTM.py
I implemented two models: Child Conv Tree LSTM and Child Sum Tree LSTM, performances will be print on the screen.

*It will be hard to run the treeLSTM parallelly in GPU, so I keep batch size equal to 1 and use CPU to run this model. Due to the size of training sample, it won't cost a lot to train the model (less than one hour for 30 epochs).*
