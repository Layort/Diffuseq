import json
# file1 = open(r"\Users\macbookxr\Documents\DiffuSeq-main\datasets\ubuntu\ubuntu_data_train.json","r")

file2 = open(r"/Users/macbookxr/Documents/DiffuSeq-main/datasets/ubuntu/ubuntu_data_test.json","r")
# file3 = open(r"\Users\macbookxr\Documents\DiffuSeq-main\datasets\ubuntu\ubuntu_data_valid.json","r")
a = [];
#
# x = 0;
# for row in file1:
#     # if(x<11000):
#     item = json.loads(row)
#     src = '\x01'.join(item["context"])
#     relation_at = item["relation_at"]
#     relation_at.append([len(relation_at)+1,item["ans_idx"]])
#     a.append({"src":src, "trg":item["answer"], "relation_at":relation_at })
#     #     x+=1;
#     # else:
#     #     break;
#
#
# file1.close();
#
# file1 = open(r"\Users\macbookxr\Documents\DiffuSeq-main\datasets\ubuntu\train.jsonl","w")
# for i in range(x):
#         file1.write("{\"src\":\"" + a[i]["src"] + "\", \"trg\":\"" + a[i]["trg"] + "\", " + "\"relation_at\":" + str(a[i]["relation_at"]) +"}\n")
# file1.close();

b = [];
x = 0;
for row in file2:
    if(x<1000):
        item = json.loads(row)
        src = ' '.join(item["context"])
        if x==0:
            print(src)
            break
        relation_at = item["relation_at"]
        relation_at.append([len(relation_at)+1,item["ans_idx"]])
        b.append({"src":src , "trg":item["answer"], "relation_at":relation_at })
        x+=1
    else:
        break

file2.close();
file2 = open(r"\Users\macbookxr\Documents\DiffuSeq-main\datasets\ubuntu\test.jsonl","w")


for i in range(x):
    file2.write("{\"src\":\""+ b[i]["src"] + "\", \"trg\":\""+ b [i]["trg"]+ "\", " + "\"relation_at\":"+str(b[i]["relation_at"]) +"}\n")
file2.close();

