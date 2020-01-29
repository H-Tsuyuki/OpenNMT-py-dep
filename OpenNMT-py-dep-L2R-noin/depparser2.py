import spacy
import time

nlp = spacy.load("en_core_web_sm")

def set_children(doc, dic, token):
    children = []
    for ch in token.children:
        if doc.index(token) > doc.index(ch):
            kid = [ch.text, ch.dep_, doc.index(ch)+1]
            children.append(kid)
    if children == []:
        children.append([token.text, "no_child", doc.index(token)+1])
    children.insert(0,len(children))

    dic["children"] = children
    return dic


def top_downize(sent):
    
    doc = nlp(sent)
    root = [token for token in doc if token.head == token]   
    root = root[0]
    doc = [i for i in doc]
    c=[{"word":"<s>", "parent":["<s>", "<s>", 0], "children":[1,["<s>", "<s>", 0]]}]

    for i, token in enumerate(doc):
        dic = {"word":token.text, "parent":[token.head.text, token.dep_, doc.index(token.head)+1]}
        if i+1< dic["parent"][2]:
            dic["parent"] = [token.text, "no_parent", doc.index(token)+1]
        dic = set_children(doc, dic, token)
        c.append(dic)
    
    dic={"word":"</s>", "parent":["</s>", "</s>", doc.index(token)+2], "children":[1,["</s>", "</s>", doc.index(token)+2]]}
    c.append(dic)
    return c

#ppp = top_downize('Those two men who are teachers have three pen.')
#for i in ppp:
#    print(i)
#from IPython.core.debugger import Pdb; Pdb().set_trace()
#print(top_downize('Lady with a green mask at the dentist and she just look very unhappy.'.lower()))

