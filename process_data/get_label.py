import json
label_file =  open('ann.json', 'r')
label_map = {}

for line in label_file:
	line = line.strip()
	label_set = json.loads(line)
	tweetid = label_set['tweetid']
	if(label_set.get("responsetype-vs-source", 0)!=0): #reply
		stance = label_set['responsetype-vs-source']
		if(stance=="agreed"):
			label_map[tweetid] = 0
		elif(stance=="disagreed"):
			label_map[tweetid] = 1
		elif(stance=="appeal-for-more-information"):
			label_map[tweetid] = 2
		elif(stance=="comment"):
			label_map[tweetid] = 3
	elif(label_set.get("support", 0)!=0): #source tweets
		stance = label_set['support']
		if(stance=="supporting"):
			label_map[tweetid] = 0
		else:
			label_map[tweetid] = 1 # denying
	else: # main tweet
		label_map[tweetid] = 4

five_events = ['ch', 'fg', 'gc', 'ow', 'ss']

for events in five_events:

	json_file = open(events + '_ori_tree.json','r')
	json_out = open(events + '_label.txt', 'w')
	for line in json_file:
		line = line.strip()
		tree_in = json.loads(line)
		for key in tree_in.keys():
			new_tree_in = {}
			new_tree_in[key] = tree_in[key] # only keep the first main tweet to keep each thread is a key
			tree_in = new_tree_in
			break
		def in_loop(current_data):
			global tree_id
			global tree_out
			if(current_data==None):
				return 0
			for new_key in current_data:
				in_loop(current_data[new_key])
				json_out.write(str(label_map.get(new_key,4)))
				json_out.write('\t')
		in_loop(tree_in)
		json_out.write('\n')
	json_out.close()
