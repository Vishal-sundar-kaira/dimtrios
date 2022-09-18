
from flask import Flask, render_template, url_for, request
import cv2
import time

app = Flask(__name__)
def face(url):
    start=time.time()
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    nurl="./static/styles/"+url
    img=cv2.imread(nurl)
    cv2.imwrite('./static/styles/oldfile.png',img)
    # IMG-20191228-WA0010.jpg,IMG-20220106-WA0019.jpg
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(grayimg,1.1,5)
    for x,y,w,h in face:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    # cv2.imshow("Gray",img)
    # end=time.time()
    # timing=end-start
    # print(f"time is: {timing}")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('./static/styles/newfile.png',img)

def summarize(text):
    import nltk
    from nltk.corpus import stopwords 
    from nltk.cluster.util import cosine_distance
    import numpy as np
    import networkx as nx

    def read_article(file_name):
        file=open(file_name,"r")
        filedata=file.readlines()
        article=filedata[0].split(". ")
        sentences=[]

        for sentence in article:
            #print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
        sentences.pop()

        return sentences
    def sentence_similarity(sent1,sent2,stopwords=None):
        if stopwords is None:
            stopwords=[]
        sent1=[w.lower() for w in sent1]
        sent2=[w.lower() for w in sent2]
        all_words=list(set(sent1 + sent2))
        
        vector1=[0]*len(all_words)
        vector2=[0]*len(all_words)
        
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)]+=1
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)]+=1
        
        return 1-cosine_distance(vector1,vector2)


    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary(file_name, top_n=5):
        nltk.download("stopwords")
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  read_article(file_name)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):
           summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize text
        print("Summarize Text: \n", ". ".join(summarize_text))
        return summarize_text

    # let's begin
    sumup=generate_summary( text, 2)
    return sumup

@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/home",methods=["GET","POST"])
def home():
    print("hello world")
    if request.method=="POST":
        url=request.form.get("url")
        face(url)
        return render_template("index.html",reimg1="oldfile.png",reimg2="newfile.png")
    return render_template("index.html")

@app.route("/text",methods=["GET","POST"])
def text():
    print("hello world")
    if request.method=="POST":
        url=request.form.get("url")
        sumupf=summarize(url)
        return render_template("index2.html",sumupf=sumupf,old="output")
    return render_template("index2.html")




    




if __name__ == "__main__":
    app.run(debug=True)