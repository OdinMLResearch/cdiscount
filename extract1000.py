import json

top1000 = {}
top1000rev = {}
with open('top1000.txt') as tf:
    lines = tf.read().split("\n")
    for i, v in enumerate(lines):
        try:
            v = int(v)
            top1000[v] = i
            top1000rev[i] = v
        except ValueError:
            print("ValueError")

with open("category2id1000.json", "w", encoding="utf8") as out:
    json.dump(top1000, out)

with open("id2category1000.json", "w", encoding="utf8") as out:
    json.dump(top1000rev, out)
