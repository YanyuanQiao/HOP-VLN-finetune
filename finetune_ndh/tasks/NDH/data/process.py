import json
import glob
from ipdb import set_trace

#test 1384 173
#val_seen 382 47.75 
#val_unseen 907 113.375

jfiles_all = '/hdd2/code/pretrain_cvdn/PREVALENT_R2R/tasks/NDH/data/val_unseen.json'
jfiles_all = '/hdd2/code/pretrain_cvdn/PREVALENT_R2R/tasks/NDH/data/val_seen.json'
jfiles_all = "/hdd2/code/pretrain_cvdn/PREVALENT_R2R/tasks/NDH/data/test.json"
with open(jfiles_all) as f:
	all_trajs = json.load(f)
instr_ids = []
for item in all_trajs:
	instr_ids.append(item['inst_idx'])
print
set_trace()
