import pickle,pprint

with open('result.dict', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.#协议版本被自动探测到并且使用，所以我们不需要明确它是什么。
    data = pickle.load(f) #使用pickle的load函数下载被打开被读取到的数据。

with open("char_dict", "rb") as f:

    data1 = pickle.load(f)
    # pprint.pprint(data1)
    # print(dir(data))
    # print(type({}))
    # print(sorted(data1.items(), key=lambda e: e[1], reverse=True))
with open("dict.txt", "w") as f:
    for key, values in sorted(data1.items(), key=lambda e: e[1], reverse=False):
        f.writelines('{0} {1} '.format(key, values) + "\n")

with open("result.txt", "w") as f:
        for i in range(len(data['prob'])):
            f.writelines('{0} {1} {2}'.format(data['prob'][i][0], data['indices'][i][0], data['groundtruth'][i])+"\n")
