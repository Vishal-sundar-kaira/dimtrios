
import re
from flask import Flask, render_template, url_for, request
import cv2
import time
import numpy as np
import pandas as pd

app = Flask(__name__)
def face(url):
    c=0
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
        c=c+1
    # cv2.imshow("Gray",img)
    # end=time.time()
    # timing=end-start
    # print(f"time is: {timing}")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('./static/styles/newfile.png',img)
    return c

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
def language(txt):
    import numpy as np
    import pandas as pd
    from flask import Flask, request, render_template
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    import re

    cv = CountVectorizer()
    le = LabelEncoder()
    df = pd.read_csv("language Detection.csv")

    X = df["Text"]
    y = df["Language"]

    y = le.fit_transform(y)

    text_list = []

    # iterating through all the text
    for text in X:         
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text) # removes all the symbols and numbers
        text = re.sub(r'[[]]', ' ', text)   
        text = text.lower()          # converts all the text to lower case
        text_list.append(text)       # appends the text to the text_list
    
    X = cv.fit_transform(text_list).toarray() 
                                            
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.80)

    
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB()
    model.fit(x_train, y_train)

    x = cv.transform([txt]).toarray()# convert text to bag of words model (Vector)
    language = model.predict(x) # predict the language
    lang = le.inverse_transform(language) # find the language corresponding with the predicted value

    output = lang[0]
    return output



#declaring global variables (are used later on)
clicked = False
r = g = b = xpos = ypos = 0

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('color_dataset.csv', names=index, header=None)

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

#function to get x,y coordinates of mouse double click
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos,clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

def processor(img):

    global clicked
    clicked = False
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_function)

    while(True):
        cv2.imshow("image", img)
        cv2.imwrite('./static/styles/oldfile1.png',img)
        if clicked:
            cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)
            text = getColorName(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
            
            #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
            cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
           
            #For very light colours we will display text in black colour
            if(r+g+b>=600):
                cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
           
            clicked=False
        #Break the loop when user hits 'esc' key 
        if cv2.waitKey(20) & 0xFF ==27:
            break
    cv2.destroyAllWindows()


#Reading the image with opencv

@app.route("/",methods=["GET","POST"])
def index():
    return render_template("homepage.html")

@app.route("/home",methods=["GET","POST"])
def home():
    return render_template("homepage.html")

@app.route("/faced",methods=["GET","POST"])
def faced():
    print("hello world")
    if request.method=="POST":
        url=request.form.get("url")
        obj=face(url)
        return render_template("index.html",reimg1="oldfile.png",reimg2="newfile.png",obj=obj)
    return render_template("index.html")

@app.route("/text",methods=["GET","POST"])
def text():
    print("hello world")
    if request.method=="POST":
        url=request.form.get("url")
        sumupf=summarize(url)
        return render_template("index2.html",sumupf=sumupf,old="output")
    return render_template("index2.html")
@app.route("/lang",methods=["GET","POST"])
def lang():
    print("Hello world")
    if request.method=="POST":
        url=request.form.get("url")
        out=language(url)
        return render_template('index4.html', prediction="language is in "+ out)
    return render_template("index4.html")
@app.route("/color",methods=["GET","POST"])
def color():
    if request.method=="POST":
        global img
        url=request.form.get("url")
        img = cv2.imread("./static/styles/"+url)
        processor(img)
        return render_template("index3.html",cdold="oldfile1.png")
    return render_template("index3.html")



    




if __name__ == "__main__":
   app.run(debug=True)
