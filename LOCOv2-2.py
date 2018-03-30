#case insensitive comparison
import time
import os

import pyautogui

import PIL.ImageGrab as IG
import ctypes

try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract
import cv2
import requests
import numpy as np
from nltk.tokenize import word_tokenize

#import asyncio
#import aiohttp
from concurrent import futures

DEBUG=1

URL = "https://www.googleapis.com/customsearch/v1"
key1='AIzaSyBZyac_0Che_oV0w_-_iRTlFT1Tx-8dECM'
key2='AIzaSyDqs69Tzx0xqQW77LhgXbgQ9_Bkmoo6qpA'
key3='AIzaSyC2KirQeyGzpWih9IUYTEooWfkwgIodf7Y'
key4='AIzaSyDA6s72rBCpck33tpjORF_vIeFL98doTxI'

cx1='007860983932491076410:npc4ahfb1bw'
cx2='005418862687413293148:tmrxys5f9ko'
cx3='006972622726393629279:brsabyap0uk'
cx4='005029266469835314094:iosyfjsigme'

KEY = key1
CX = cx1

def Timelapse(str,flag=0):
    if(flag==2):
        Timelapse.t=time.time()
        Timelapse.beg=time.time()
        return
    if(flag==1):
        print(str," : since beginning : ",time.time() -Timelapse.beg)

    if(DEBUG):
        print(str,": ",time.time() -Timelapse.t)
    Timelapse.t=time.time()


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

def ExtractQ():
    
    pyautogui.click((1142, 600))
    #time.sleep(0.5)


    FILEPATH1 = "C:\\Users\\Yash Kamal\\Downloads\\"
    FILEPATH2 = "C:\\Users\\Yash Kamal\\Desktop\\loco\\"
    WORD1 = 'screenshot'
    WORD2 = 'Capture'
    
    KEYWORD = WORD1
    FILEPATH = FILEPATH1
    filepath = FILEPATH +"*.jpg"

    #check if file with filepath exists
    def Filexist():
        cnt=5
        textfile=FILEPATH+"screenshot.jpg"

        while(not os.path.exists(textfile) and  cnt):    
            time.sleep(0.5)
            cnt=cnt-1
        if(cnt):
            return textfile
        print("Time out:Could not find the screenshot!!!!!!!!!!!!!!")
        return "timeout"
        
    Textfile = Filexist()
    if(Textfile=="timeout"):
        return []
    
    if(DEBUG):
        print("^^",Textfile)

    #Textfile_n = Textfile
    Textfile_n = FILEPATH+"Done"+str(time.time())+".jpg"
    os.rename(Textfile,Textfile_n)
    
    
    #Timelapse("Checkpoint")
    X= (pytesseract.image_to_string(Image.open(Textfile_n)))

    #cv2.imshow("fsdg",image[350:350+200, 50:50+450])
    #cv2.waitKey(0)
    #X= (pytesseract.image_to_string(image[350:350+200, 50:50+450]))
    #X = pytesseract.image_to_string(image)
    #Timelapse("OCR scan!")
    
    if(DEBUG):
        print(X)
    
    lis= X.split('\n')
    flag=0
    ques="";
    optionlis=[]
    for e in lis:
        if(flag ==1 or len(e) > 12 or ('?' in e)):
            flag=1
            ques+=" "+e
            if('?' in e):
                flag=2
                break;
        
    if(flag != 2):
        print("WARNING!!!!!!!!!! Question may have not been extracted properly")
        
    if(DEBUG):
        print('----------------------------------------------------------')
        print(ques)

    return [ques,Textfile_n]


def ExtractOPT(path):
    
    image = cv2.imread(path)

    if(DEBUG):
        #cv2.imshow("fbbgd",image)
        print(len(image),len(image[0]))

    fy= len(image)/960.0
    fx= len(image[0])/540.0

    S_HEIGHT = int(330*fy)
    S_WIDTH =  int(20*fx)
    S_X1 =     int(140*fx)
    S_X2 =     int(400*fx)
    S_Y =      int(450*fy)
    ROOM =     int(10*fy)
    OPTX1 =    int(75*fx)
    OPTX2 =    int(375*fx)
    Linewidth= int(3*fy)
    
    
    strip1 = cv2.cvtColor(image[S_Y:S_Y+S_HEIGHT, S_X1:S_X1+S_WIDTH],cv2.COLOR_BGR2GRAY)
    strip2 = cv2.cvtColor(image[S_Y:S_Y+S_HEIGHT, S_X2:S_X2+S_WIDTH],cv2.COLOR_BGR2GRAY)  

    '''
    cv2.imshow("strip1", strip1)
    cv2.waitKey(0)
    cv2.imshow("strip2", strip2)
    cv2.waitKey(0)
    '''


    # function to check if row is white( detect boundary)
    def F1(row):
        cnt=0
        for e in row:
            if(e > 80 or e < 65):
                cnt=cnt+1
        cnt=cnt/(1.0 * len(row))
        if(cnt > 0.95):
            return 1
        return 0

    Line=[]
    i=1
    cnt=0
    for row1,row2 in zip(strip1,strip2):
        a1=F1(row1)
        a2=F1(row2)
        if(a1==1 and a2==1):
            cnt=cnt+1
        elif(cnt!=0):
            if(cnt>= Linewidth ):
                Line.append(i)
            cnt=0
        i=i+1
    
    #print(Line)
    if(len(Line) <6):
        print("!!!!!!!!!!!!!!!!!!!couldn't read all option boundaries")
        return []


    def func(ind=0):
        opt1 = image[ S_Y+Line[ind]+ROOM:S_Y+Line[ind+1]-ROOM,OPTX1:OPTX2]
        opt2 = image[ S_Y+Line[ind+2]+ROOM:S_Y+Line[ind+3]-ROOM,OPTX1:OPTX2]
        opt3 = image[ S_Y+Line[ind+4]+ROOM:S_Y+Line[ind+5]-ROOM,OPTX1:OPTX2]
        opt1=np.concatenate((opt1,opt2))
        opt1=np.concatenate((opt1,opt3))

        
        #cv2.imshow("strip2", opt1)
        #cv2.waitKey(0)
        #filename = "test001.png"
        #cv2.imwrite(filename, opt1)
        #Timelapse("Checkpoint")

        
        X = pytesseract.image_to_string(opt1, lang='eng', boxes=False, \
                     config='--psm 6 --eom 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,-')

        #Timelapse("OCR scan")

        #X= (pytesseract.image_to_string(Image.open('test001.png')))
        if(DEBUG):
            print(X)
        cv2.waitKey(0)
        lis= X.split('\n')
        optionlis=[]
        for e in lis:
            if( len(e) >=1):
                optionlis.append(e)
        return optionlis

    opts=func()
    if(DEBUG):
        print("options list:",opts)

    if(len(opts)<3):
        print("!!!!!!!!!!!!!!!!!!!couldn't extract enough options: ",opts)
        return []
    
    opt1=opts[0]
    opt2=opts[1]
    opt3=opts[2]
    if(DEBUG):
        print('----------------------------------------------------------')
        print(opt1,opt2,opt3)

    return [opt1,opt2,opt3]
    

stop_words = {'ourselves', 'hers', 'between','?','!','.',',', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
def PreProcess(ques):
    word_tokens = word_tokenize(ques)
    #word_tokens = ques.split()
    ques = [w for w in word_tokens if not w in stop_words]
    return ques

def F(text,ques):
    score=0.0
    Txt=text.split()
    for word in ques :
        if(word in Txt):
            score+=1.0
    score=score/(1.0 * len(ques) )
    return score

def Present(opt,text):
    room=0
    if(len(opt)>5):
        room=2
    if(len(opt)>2):
        room=1
    for s in [text[i:i+len(opt)] for i in range(0,len(text)-len(opt)+1)]:
        if sum(a!=b for a,b in zip(opt,s))<=room:
            return True
    return False

def Present2(opt,text,ff=0):
    opt=opt.lower()
    text=text.lower()

    if(Present(opt,text)):
        #return 1.5+ff
        return 1.2
    
    score=0
    cnt=0
    print(opt)
    for word in opt.split():    
        word=word.rstrip(',')
        #print(word)
        if(word not in stop_words):
            
            if(len(word)<2):
                cnt=cnt+0.1
                continue
            if(Present(word,text)):
               score=score+1
            cnt=cnt+1
    if(cnt):        
        score=score/(1.0*cnt)
    return score

def BiasedScore(opt,ques,ques1):
    query = ques+" "+opt
    PARAMS = {'key':KEY,'cx':CX,'q':query}    

    if(DEBUG):
        print('query: ',query)

    r = requests.get(url = URL, params = PARAMS).json()
    '''
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, params=PARAMS) as resp:
            r = await resp.json()
    '''
    score=0.0
    if('items' in r):
        for item in r['items']:
            sc=Present2(opt, item['snippet'])
            if(sc>0.01):
                score+=sc*F(item['snippet'],ques1)
        score=score/(0.5*len(r['items']))
        if(DEBUG):
            print("biased search yield: ",len(r["items"]))
    if(DEBUG):
        print("score%%%%%",score)
    return score

def Unbiasedscore2(opt1,opt2,opt3,ques,flag):
    if(flag==0):
        return [0,0,0]
    
    GAIN=1.5
    query = ques
    PARAMS = {'key':KEY,'cx':CX,'q':query}    
    r = requests.get(url = URL, params = PARAMS).json()

    '''
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, params=PARAMS) as resp:
            r = await resp.json()
    '''
    
    score1=0
    score2=0
    score3=0
    if('items' in r):   
        for item in r['items']:
            score1+=GAIN*Present2(opt1,item['snippet'],1)   
            score2+=GAIN*Present2(opt2,item['snippet'],1)
            score3+=GAIN*Present2(opt3,item['snippet'],1)
            
        print(len(r['items']))
        score1=score1/(1.0*len(r['items']))
        score2=score2/(1.0*len(r['items']))
        score3=score3/(1.0*len(r['items']))
        if(DEBUG):
            print("unbiased search2 yield: ",len(r["items"]))
    else:
        print("something is wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    if(DEBUG):
        print(score1,"  -- ",score2," --  ",score3)
    return [score1,score2,score3]


def Unbiasedscore0(ques1):
    ques = " ".join(ques1)
    query = ques
    PARAMS = {'key':KEY,'cx':CX,'q':query}    
    r = requests.get(url = URL, params = PARAMS).json()
    return [r,query]


def Unbiasedscore(opt1,opt2,opt3,r,query):
    #print(r)
    GAIN=1.5
    score1=0
    score2=0
    score3=0
    flag=0
    if('items' in r):   
        for item in r['items']:
            #print(item['snippet'])
            score1+=GAIN*Present2(opt1,item['snippet'],1)   
            score2+=GAIN*Present2(opt2,item['snippet'],1)
            score3+=GAIN*Present2(opt3,item['snippet'],1)
            
        score1=score1/(1.0*len(r['items']))
        score2=score2/(1.0*len(r['items']))
        score3=score3/(1.0*len(r['items']))
        if(DEBUG):
            print("unbiased search yield: ",len(r["items"]))
    else:
        flag=1

    if('spelling' in r):
        query=r['spelling']['correctedQuery']
    if(DEBUG):
        print(score1,"    ",score2,"   ",score3)
        
    return [score1,score2,score3,query,flag]

    
def Search():
    Timelapse("reset",2)
    
    _ =ExtractQ()
    if(not _):
        print("couldn't extract question !!!!!!!!!!!!!!!")
        return
    ques,PATH =_
    ques1=PreProcess(ques)

    QTYPE = 1
    if("NOT" in ques1):
        QTYPE = -1
    
    Timelapse("question extracted")

    
    with futures.ThreadPoolExecutor() as executer:
        job1 = executer.submit(Unbiasedscore0,ques1)
        job2 = executer.submit(ExtractOPT,PATH)
        futures.wait([job1,job2],None,futures.ALL_COMPLETED)
        res1 = job1.result()
        res2 = job2.result()
        
    '''
    res1 = Unbiasedscore0(ques1)
    res2 = ExtractOPT(PATH)
    '''

    Timelapse("Stage1 cleared")
    #return

    if(not res2):
        print("couldn't extract options!!!!!!!!!!!!!!!")
        return
    opt1,opt2,opt3 =res2
    
       

    if(DEBUG):
        Timelapse("question preprocessing complete!")
        print("after processing question: ",ques1)

    _ = Unbiasedscore(opt1,opt2,opt3,res1[0],res1[1])
    
    score1=_[0]
    score2=_[1]
    score3=_[2]
    
    if(DEBUG):
        Timelapse("Unbiased search complete!")

    
    with futures.ThreadPoolExecutor() as executer:
        job1 = executer.submit(BiasedScore,opt1,_[3],ques1)
        job2 = executer.submit(BiasedScore,opt2,_[3],ques1)
        job3 = executer.submit(BiasedScore,opt3,_[3],ques1)
        job4 = executer.submit(Unbiasedscore2,opt1,opt2,opt3,_[3],_[4])
        futures.wait([job1,job2,job3,job4],None,futures.ALL_COMPLETED)
        scor1 = job1.result()
        scor2 = job2.result()
        scor3 = job3.result()
        _ = job4.result()
    
    '''    
    scor1 = BiasedScore(opt1,_[3],ques1)
    scor2 = BiasedScore(opt2,_[3],ques1)
    scor3 = BiasedScore(opt3,_[3],ques1)
    _  = Unbiasedscore2(opt1,opt2,opt3,_[3],_[4])
    '''
    #scor1,scor2,scor3,_ = await asyncio.gather(BiasedScore(opt1,_[3],ques1),BiasedScore(opt2,_[3],ques1),
    
    
    score1+=_[0]+scor1
    score2+=_[1]+scor2
    score3+=_[2]+scor3

    if(DEBUG):
        Timelapse("Biased search complete")
        
    print("******************************************************************")
    print(ques)
    print(opt1,score1)
    print(opt2,score2)
    print(opt3,score3)
    print("*****************************************************************")

    score1 = score1*QTYPE
    score2 = score2*QTYPE
    score3 = score3*QTYPE

    if(QTYPE ==-1):
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<INVERSE OUTPUT!!!>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #ctypes.windll.user32.MessageBoxW(0, "aaaaa", "answer", 1)
    if(score1> score2 and score1 >score3):
        print(opt1,"\n",opt1,"\n",opt1,"\n",opt1)
    if(score2> score1 and score2 >score3):
        print(opt2,"\n",opt2,"\n",opt2,"\n",opt2)
    
    if(score3> score1 and score3 >score2):
        print(opt3,"\n",opt3,"\n",opt3,"\n",opt3)
    
    Timelapse("Execution complete in!",1)
    f=open("filedata1.txt","r")
    A=int(f.read())
    f.close()

    f=open("filedata1.txt","w")
    f.write(str(A+4))
    print("requests remaining: ",96-A)
    f.close()


'''
def Main():
    Timelapse("reset",2)
    
    _ =ExtractQ()
    if(not _):
        print("couldn't extract question !!!!!!!!!!!!!!!")
        return
    ques,PATH =_

              
    _ = ExtractOPT(PATH)
    if(not _):
        print("couldn't extract options!!!!!!!!!!!!!!!")
        return
    opt1,opt2,opt3 =_

    ques =_[0]
    opt1 =_[1]
    opt2 =_[2]
    opt3 =_[3]
    
    if(DEBUG):
        Timelapse("Extraction complete in ",1)
    #Search(ques,opt1,opt2,opt3)
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(Search(ques,opt1,opt2,opt3))
    #loop.close()
'''
    
    
Timelapse.t=time.time()
Timelapse.beg=time.time()
inp='Z'
while(inp!='n'):
    inp =input("prompt")
    if(inp=='y'):
        Search()
    








