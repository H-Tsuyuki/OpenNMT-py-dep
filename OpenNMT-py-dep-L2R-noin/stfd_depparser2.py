import stfd_dep

def set_children(doc, dic, token):
    children = []
    for ch in token.children:
        if doc.index(token) > doc.index(ch):
            kid = [ch.text.text, ch.dep_, doc.index(ch)+1]
            children.append(kid)
    if children == []:
        children.append([token.text.text, "no_child", doc.index(token)+1])
    children.insert(0,len(children))

    dic["children"] = children
    return dic


def top_downize(sent):
    
    doc = stfd_dep.stfd_depparse(sent)
#    root = [token for token in doc if token.head == 'ROOT']   
#    root = root[0]
#    doc = [i for i in doc]
    c=[{"word":"<s>", "parent":["<s>", "<s>", 0], "children":[1,["<s>", "<s>", 0]]}]

    for i, token in enumerate(doc):
        dic = {"word":token.text.text, "parent":[token.head.text.text, token.dep_, doc.index(token.head)+1]}
        if i+1< dic["parent"][2]:
            dic["parent"] = [token.text.text, "no_parent", doc.index(token)+1]
        dic = set_children(doc, dic, token)
        c.append(dic)
    
    dic={"word":"</s>", "parent":["</s>", "</s>", doc.index(token)+2], "children":[1,["</s>", "</s>", doc.index(token)+2]]}
    c.append(dic)
    return c

#ppp = top_downize('Those two men who are teachers have three pen.')
#ppp = top_downize('a black dog and a spotted dog are fighting.')
#for i in ppp:
#    print(i)
#print(top_downize('Lady with a green mask at the dentist and she just look very unhappy.'.lower()))

