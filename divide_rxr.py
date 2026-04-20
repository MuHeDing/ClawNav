import json

data_path=r'/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/rxr/val_unseen/val_unseen_guide_en.json'


val_unseen_guide1={"episodes":[]}

val_unseen_guide2={"episodes":[]}

with open(data_path, 'r') as f:
    data= json.load(f)
    print(data.keys())
    print(data['episodes'][0])
    print(len(data['episodes'])) # 3669
    exit()
    
    for i,k in enumerate(data['episodes']):
        if i<10:
            val_unseen_guide1['episodes'].append(k)
        else:
            val_unseen_guide2['episodes'].append(k)

print(len(val_unseen_guide1['episodes']))

# print(val_unseen_guide1['episodes'][0])

print(len(val_unseen_guide2['episodes']))

with open('/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/rxr/val_unseen/val_unseen_guide_en1.json', "w", encoding="utf-8") as f:
    json.dump(val_unseen_guide1, f, ensure_ascii=False, sort_keys=False)


with open('/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/rxr/val_unseen/val_unseen_guide_en2.json', "w", encoding="utf-8") as f:
    json.dump(val_unseen_guide2, f, ensure_ascii=False, sort_keys=False)


with open('/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/rxr/val_unseen/val_unseen_guide_en1_indent4.json', "w", encoding="utf-8") as f:
    json.dump(val_unseen_guide1, f, indent=4, ensure_ascii=False, sort_keys=False)


with open('/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/rxr/val_unseen/val_unseen_guide_en2_indent4.json', "w", encoding="utf-8") as f:
    json.dump(val_unseen_guide2, f, indent=4, ensure_ascii=False, sort_keys=False)

