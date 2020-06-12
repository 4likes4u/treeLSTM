from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
import configuration
import encoder_manager
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Set paths to the model.
# 
VOCAB_FILE = "pretrained/uni/vocab.txt"
EMBEDDING_MATRIX_FILE = "pretrained/uni/embeddings.npy"
CHECKPOINT_PATH = "pretrained/uni/model.ckpt-501424"

# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)
VOCAB_FILE = "pretrained/bi/vocab.txt"
EMBEDDING_MATRIX_FILE = "pretrained/bi/embeddings.npy"
CHECKPOINT_PATH = "pretrained/bi/model.ckpt-500008"

encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

five_events = ['ch', 'fg', 'gc', 'ow', 'ss']
stat = {}
stat['ch'] = (74, 93)
stat['fg'] = (46, 95)
stat['gc'] = (25, 43)
stat['ow'] = (58, 46)
stat['ss'] = (71, 103)

for events in five_events:
	threads_num = stat[events][0]
	max_tweets = 103
	matrix = np.zeros((threads_num, max_tweets, 4800)) #totally 157 threads, padding to 95 tweets, each tweets 4800d
	text_in = open(events + '_tweets.txt','r')
	thread_index = 0
	for thread in text_in:
		thread = thread.strip()
		thread = thread.split('\t')
		thread = thread + ["padding_tweets"]*(max_tweets-len(thread))
		embedding = encoder.encode(thread)
		embedding = embedding.reshape(1,max_tweets,4800)
		matrix[thread_index]= embedding
		thread_index = thread_index + 1
	print(str(thread_index))
	np.save(events + "v2.npy", matrix)
