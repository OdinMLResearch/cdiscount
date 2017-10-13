import json
import bson

f = open('/datadrive/Cdiscount/train.bson', 'rb')

def create_category_map():
        print("Create Id Category Map ....")
        Id2CategoryMap = {}
        Category2IdMap = {}
        dedupSet = set()
        
        data = bson.decode_file_iter(f)
        delayed_load = []

        count = 0
        i = 0
        try:
            for c, d in enumerate(data):
                c = c + 1
                if c%100 == 0:
                    print("Start reading %s record. Num of category %s" % (c, i))
                if c> 1000:
                    pass
                    #break
                target = d['category_id']
                if target not in dedupSet:
                    dedupSet.add(target)
                    Id2CategoryMap[i] = target
                    Category2IdMap[target] = i
                    i = i + 1

        except IndexError:
            pass;

        print("Total num of category: %d\n", len (dedupSet))

        return Id2CategoryMap, Category2IdMap

id2category, category2id = create_category_map()

f.close()

with open("id2category.json", "w", encoding="utf8") as out:
    json.dump(id2category, out)


with open("category2id.json", "w", encoding="utf8") as out:
    json.dump(category2id, out)
