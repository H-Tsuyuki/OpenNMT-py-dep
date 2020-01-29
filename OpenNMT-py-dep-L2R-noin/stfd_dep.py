import spacy
import stanfordnlp

nlp = stanfordnlp.Pipeline()
class SubTree(object):
    def __init__(self, word, label, head):
        self.text = word
        self.head = head
        self.dep_ = label
        self.children = []
        self.layer = 0

    def set_ch(self, ch):
        self.children.append(ch)

    def set_head(self, head):
        self.head = head 

def stfd_depparse(sent):
    doc = nlp(sent)
    aaa = [i for i in doc.sentences[0].dependencies]
#    print(doc.sentences[0].print_dependencies())
    sbts = []
    for i in aaa:
        sbt = SubTree(i[2], i[1], i[0])
        #sbt = SubTree(i[2].text, i[1], i[0].text)
        sbts.append(sbt)

    for st in sbts:
        [(st.set_ch(sbt), sbt.set_head(st)) for sbt in sbts if st.text == sbt.head and st.text!=sbt.text]
        if st.head.text == 'ROOT':
            st.head = st
    return sbts

#sent = "Those two men are teachers."
#sent = "a black dog and a spotted dog are fighting."
#doc1 = stfd_depparse(sent)

#nlp_spacy = spacy.load("en_core_web_sm")
#doc2 = nlp_spacy(sent)


