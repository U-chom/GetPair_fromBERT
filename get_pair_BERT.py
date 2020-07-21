#D(u)とD(v)に出現するSVMscoreTOP100の単語を与えた場合，D(u)の単語とD(v)の単語の類似度を図り，それがお互いに最短距離のペアを抽出する．
#その結果として，対照語の抽出を狙う．
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import glob
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pyknp import Juman
import csv
import pprint
from itertools import zip_longest
import datetime
dev_gpu0 = torch.device('cuda:0')
dev_gpu1 = torch.device('cuda:1')

jumanpp = Juman()
model = BertModel.from_pretrained("./model/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM/")
bert_tokenizer = BertTokenizer("./model/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM/vocab.txt",
                               do_lower_case=False, do_basic_tokenize=False)

# lim = 1000
lim2 = 0

class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]

juman_tokenizer = JumanTokenizer()

def stopword(text):
    text = text.replace(")","")
    text = text.replace("(","")
    text = text.replace("'","")
    text = text.replace('"',"")
    text = text.replace("、","")
    text = text.replace("。","")
    text = text.replace(",","")
    text = text.replace('\\',"")
    text = text.replace("[","")
    text = text.replace("]","")
    text = text.replace("「","")
    text = text.replace("」","")
    text = text.replace("#","")
    text = text.replace("?","")
    text = text.replace("!","")
    text = text.replace(";","")
    text = text.replace(":","")
    text = text.replace("：","")
    text = text.replace("*","")
    text = text.replace("@","")
    text = text.replace("\n","")
    text = text.replace("-","")
    text = text.replace("_","")
    text = text.replace("『","")
    text = text.replace("』","")
    text = text.replace("\u3000","")
    return text

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def ruizi(model,p,n):
    while True:
        time = datetime.datetime.now()
        time_sta = time.time()
        
        #bert-------
        text = p
        tokens = juman_tokenizer.tokenize(text)
        bert_tokens = bert_tokenizer.tokenize(" ".join(tokens))
        ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])
        tokens_tensor = torch.tensor(ids).reshape(1, -1)
        model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = model(tokens_tensor)
        pooling_layer = -2
        Pembedding = all_encoder_layers[pooling_layer].numpy()[0]
        np.mean(Pembedding, axis=0)
        #bert-------
        text = n
        tokens = juman_tokenizer.tokenize(text)
        bert_tokens = bert_tokenizer.tokenize(" ".join(tokens))
        ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])
        tokens_tensor = torch.tensor(ids).reshape(1, -1)
        model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = model(tokens_tensor)
        pooling_layer = -2
        Nembedding = all_encoder_layers[pooling_layer].numpy()[0]
        np.mean(Nembedding, axis=0)

        similarity = cos_sim(Pembedding[1],Nembedding[1])
        break
    return similarity



def fase1(name,Rswich):
    with open("./input/"+str(name)+".txt") as f:
        svmscore = f.readlines()
        svmscore.pop(0)
        p = []
        n = []
        numkey = [0,0]
        for m in svmscore:
            m = m.replace("\n","")
            line = m.split(" ")
            line2 = [a for a in line if a != '']
            try:
                if "0.0000" in line2[1]:
                    numkey.insert(0,1)
                if numkey[0] == 0:
                    try:
                        p.append(line2[4])
                    except:
                        pass
                if numkey[0] == 1 and numkey[1] == 0:
                    if not "0.0000" in line2[1]:
                        numkey.insert(1,1)
                if numkey[0] == 1 and numkey[1] == 1:
                    try:
                        n.append(line2[4])
                    except:
                        pass
            except:
                pass

        for i,m in enumerate(p[:]):
            m = stopword(m)
            if not len(m) == 0:
                pass
            else:
                try:
                    del p[i]
                except:
                    pass
        for i,m in enumerate(n[:]):
            m = stopword(m)
            if not len(m) == 0:
                pass
            else:
                try:
                    del p[i]
                except:
                    pass
        # print(p,n)
        print("fase1:END")#ここまでは変更なし
        if Rswich == 1:##ここ変えんな
            random.shuffle(p)
            random.shuffle(n)
    return p,n

def fase2(pos,neg,lim2):
    toppp = []
    for m in pos:
        top = {}
        for n in neg:
            try:
                val = ruizi(model,m,n)
            except:
                print("だめだこりゃ")
                continue
            top[n]=val
        top = sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]
        topp = [m,top]
        toppp.append(topp)
    topnn = []
    for m in neg:
        top = {}
        for n in pos:
            try:
                val = ruizi(model,m,n)
            except:
                print("だめだこりゃ")
                continue
            top[n]=val
        top = sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]
        topn = [m,top]
        topnn.append(topn)
    print("fase2:END")
    return toppp,topnn    

def fase3(pos,neg,Rswich,gift):
    #ここで比べる．両方にとっての一番かどうか．一番でない場合は，出てきている単語をみせる．
    ave = []
    for i in pos:#[[m,[('想像', 0.6579141), ('変えれ', 0.6203517), ('現れる', 0.58531106), ('取り出す', 0.52433646)]],....]
        for j in neg:
            for li,ii in enumerate(i[1]):
                if ii[0] == j[0]:#iの５い以内にjがいる
                    # print(f"{i[0]}=>{ii[0]}")
                    for lj,jj in enumerate(j[1]):#jの５い以内にiがいる
                        if jj[0] == i[0]:
                            one_row = f"{i[0]}=>{ii[0]}:順位は{li+1}位と{lj+1}位でした=={ii[1]}"
                            gift.append(one_row)
                            ave.append(ii[1])
                            # print(f"{i[0]}=>{ii[0]} : 順位は{li+1}位と{lj+1}位でした")
                            break
                    
                        if lj == lim2:
                            break 
                if li == lim2:
                    break
    avesum = f"ここの類似度の平均={sum(ave)/len(ave)}"
    gift.append(avesum)
    
    # ここからD（v）    
    # for i in neg:#[[m,[('想像', 0.6579141), ('変えれ', 0.6203517), ('現れる', 0.58531106), ('取り出す', 0.52433646)]],....]
    #     for j in pos:
    #         for li,ii in enumerate(i[1]):
    #             if ii[0] == j[0]:#iの５い以内にjがいる
    #                 # print(f"{i[0]}=>{ii[0]}")
    #                 for lj,jj in enumerate(j[1]):#jの５い以内にiがいる
    #                     if jj[0] == i[0]:
    #                         print(f"{i[0]}=>{ii[0]} : 順位は{li+1}位と{lj+1}位でした")
    #                         break
    #                         # print(f"{ii[0]},{jj[0]}")
                    
    #                     if lj == lim2:
    #                         break 
    #             if li == lim2:
    #                 break              
    print("fase3:END")
                
FS = 12367
FSnum = [20,30]#,500,1000,5000]
textlist = ["test"]#,"1x3","1x4","2x3","2x4","3x4"]
Rswich = 0 #0:normal 1:random
u0 = []
u1 = []
u2 = []
u3 = []
u4 = []
u5 = []

for namunm,name in enumerate(textlist):
    p,n = fase1(name,Rswich)
    print(f"元の単語数=>{len(p)}")
    for Fnum in FSnum:
        pp = p[:Fnum]
        nn = n[:Fnum]
        print(f"削減された単語数=>{len(pp)}")
        dp,dn = fase2(pp,nn,lim2)
        if namunm == 0:
            u0.append(Fnum)
            fase3(dp,dn,Rswich,u0)
        elif namunm == 1:
            u1.append(Fnum)
            fase3(dp,dn,Rswich,u1)
        elif namunm == 2:
            u2.append(Fnum)
            fase3(dp,dn,Rswich,u2)
        elif namunm == 3:
            u3.append(Fnum)
            fase3(dp,dn,Rswich,u3)
        elif namunm == 4:
            u4.append(Fnum)
            fase3(dp,dn,Rswich,u4)
        elif namunm == 5:
            u5.append(Fnum)
            fase3(dp,dn,Rswich,u5)
        print(f"{Fnum}:END")

if Rswich == 0:
    filename = "./output/"+str(FS)+"BERT.csv"
elif Rswich == 1:
    filename = "./output/"+str(FS)+"RBERT.csv"
with open(filename, 'w') as f:
    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
    for u00,u11,u22,u33,u44,u55 in zip_longest(u0,u1,u2,u3,u4,u5):
        gotoCSV = [u00,u11,u22,u33,u44,u55]
        writer.writerow(gotoCSV)

# for namunm,name in enumerate(textlist):
#     p,n = fase1(name,Rswich)
#     print(f"元の単語数=>{len(p)}")
#     Fnum = 100
#     pp = p[:Fnum]
#     nn = n[:Fnum]
#     print(f"削減された単語数=>{len(pp)}")
#     dp,dn = fase2(pp,nn,lim2)
#     if namunm == 0:
#         u0.append(Fnum)
#         fase3(dp,dn,Rswich,u0)
#     elif namunm == 1:
#         u1.append(Fnum)
#         fase3(dp,dn,Rswich,u1)
#     elif namunm == 2:
#         u2.append(Fnum)
#         fase3(dp,dn,Rswich,u2)
#     elif namunm == 3:
#         u3.append(Fnum)
#         fase3(dp,dn,Rswich,u3)
#     elif namunm == 4:
#         u4.append(Fnum)
#         fase3(dp,dn,Rswich,u4)
#     elif namunm == 5:
#         u5.append(Fnum)
#         fase3(dp,dn,Rswich,u5)
#     print(f"{Fnum}:END")

# if Rswich == 0:
#     filename = "/home/yuki/Desktop/xx/"+str(FS)+"BERT.csv"
# elif Rswich == 1:
#     filename = "/home/yuki/Desktop/xx/"+str(FS)+"RBERT.csv"
# with open(filename, 'w') as f:
#     writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
#     for u00,u11,u22,u33,u44,u55 in zip_longest(u0,u1,u2,u3,u4,u5):
#         gotoCSV = [u00,u11,u22,u33,u44,u55]
#         writer.writerow(gotoCSV)