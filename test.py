import pickle as pkl
from Queue import Queue
def treeIterator(node):
    curr = node
    q = Queue()
    q.put((node,0))
    while q.qsize()>0:
        for _ in range(q.qsize()):
            temp, level = q.get()
            print "level : value" + str(level) + " : " + str(temp.value)
            if temp.data !='T' and temp.data !='F':
                for t in temp.nodes:
                    if t!= -1 :
                        q.put((t, level +1))
        # print ("\n")
        #else:
        #    break

class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

root = pkl.load(open('tree.pkl','r'))
treeIterator(root)
