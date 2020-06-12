# -*- coding: UTF-8 -*-
# there're some threads contains more than one main tweets, so we fix it here, truncate the following threads only keep the every first one.
import json
import os
import numpy as np

five_folders = ['ch', 'fg', 'gc', 'ow', 'ss']

for main_folders in five_folders:
	text_out = open(main_folders + '_tweets.txt', 'w', encoding='UTF-8')
	json_tree_out = open(main_folders + '_tree.json', 'w')
	ori_json_out = open(main_folders + '_ori_tree.json', 'w')
	max_tweets = 0
	folder_count = 0
	for folders in os.listdir(main_folders):
		if(folders.isdigit()): #ignore other files we created
			with open(main_folders +"/" + folders + "/structure.json", 'r') as f:
				tree_in = json.load(f)
			for key in tree_in.keys():
				new_tree_in = {}
				new_tree_in[key] = tree_in[key] # only keep the first main tweet to keep each thread is a key
				tree_in = new_tree_in
				break
			tree_out = str(json.dumps(tree_in)) # change type to str
			tree_id = 0
			for key in tree_in:
				tree_out = tree_out.replace(str(key),str(tree_id))
				# each tree only has one key, the source twitter:
				with open(main_folders +"/" + folders + '/source-tweets/'+str(key) + '.json', 'r') as f:
					twitter_text = json.load(f)
					text = twitter_text['text'].strip().replace('\n','').replace('\r','')
				text_out.write(text)
				text_out.write('\t')
				break # there has a bug exits 552816020403269632 file

			def in_loop(current_data):
				global tree_id
				global tree_out
				if(current_data==None):
					return 0
				for new_key in current_data:
					tree_id = tree_id + 1 
					tree_out = tree_out.replace(str(new_key), str(tree_id))
					with open(main_folders +"/" + folders + '/reactions/'+str(new_key) + '.json', 'r') as f:
						twitter_text = json.load(f)
						text = twitter_text['text'].strip().replace('\n','').replace('\r','')
					text_out.write(text)
					text_out.write('\t') #use tab to sperate the intra tweets
					in_loop(current_data[new_key])

			in_loop(tree_in[key])
			text_out.write('\n') # change line to sperate the inter tweets (i.e., different threads)
			folder_count = folder_count + 1
			json_tree_out.write(tree_out) # save string to json file
			json_tree_out.write("\n") # change_line
			ori_json_out.write(str(json.dumps(tree_in)))
			ori_json_out.write("\n")
			tree_out = json.loads(tree_out)
			if(tree_id>max_tweets):
				max_tweets = tree_id+1 #record max sentence for padding
	text_out.close()
	json_tree_out.close()
	print(main_folders + " max tweets: " + str(max_tweets))
	print(main_folders + " max threads count: " + str(folder_count))