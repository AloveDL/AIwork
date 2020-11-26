import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

def split():
    fr = open(r'data/text.txt')
    listWm = [inst.strip().split(',')[1:] for inst in fr.readlines()]
    features = listWm[0][:-1]
    for index,feature in enumerate(features):

        plt.title(feature+"划分")
        mydict = {}
        for s in listWm[1:]:
            print(s[index])
            if s[index] not in mydict.keys():
                mydict[s[index]] = 1
            else:
                mydict[s[index]] += 1
        x = range(len(mydict.keys()))
        y = list(map(int,mydict.values()))
        print(mydict.keys())
        plt.bar(x,y,align="center",color="b",tick_label=list(mydict.keys()))
        plt.xlabel("属性"+feature)
        plt.ylabel("数量")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    split()