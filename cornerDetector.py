import cv2
import numpy as np

# corner index
i=1

# corner-cropped images in numpy array
a1 = np.zeros((7,7))
a2 = np.zeros((7,7))
a3 = np.zeros((7,7))
a4 = np.zeros((7,7))
b1 = np.zeros((7,7))
b2 = np.zeros((7,7))
b3 = np.zeros((7,7))
b4 = np.zeros((7,7))

def pickPoint(event, x, y, flags, img): # picking Point - when mouse clicked
    global i
    h, w = img.shape
    if event == cv2.EVENT_LBUTTONDBLCLK:
        xText,yText =textLocation(x,y,h,w)
        if i<5:
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), color=(0, 0, 0),thickness=2)  # drawing rectangle on the image
            cv2.putText(img,'1-'+str(i),(xText,yText),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)
            saveCorner(img,x,y,i,h,w)
        elif i>=5 and i<9:
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), color=(0, 0, 0),thickness=2)  # drawing rectangle on the image
            cv2.putText(img,'2-'+str(i-4),(xText,yText),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)
            saveCorner(img,x,y,i,h,w)

        i += 1

def textLocation(x,y,h,w): # setting text location (to avoid overlapping)
    if x+70 < w and y-40>0: # default: top right
        xText = x+5
        yText = y-5
    else:
        if x-70>0 and y+30<h: # bottom left
            xText = x-70
            yText = y+30
        else:
            if x-70>0 and y-40>0: # top left
                xText = x-70
                yText = y-5
            else: # bottom right
                xText = x+5
                yText = y+30

    return xText,yText


def saveCorner(img,x,y,i,h,w): # save cropped corner images
    global a1,a2,a3,a4,b1,b2,b3,b4
    pointX ,pointY = x, y
    if pointX<3:
        pointX=3
    if pointY<3:
        pointY=3
    if pointX+4>w:
        pointX-=4
    if pointY+4>h:
        pointY-=4

    if i==1:
        a1 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==2:
        a2 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==3:
        a3 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==4:
        a4 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==5:
        b1 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==6:
        b2 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==7:
        b3 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()
    elif i==8:
        b4 = img[pointY-3:pointY+4,pointX-3:pointX+4].copy()


def cornerDetect(): # main function for corner detecting
    img1path = './imgs/1st.jpg'
    img2path = './imgs/2nd.jpg'

    img1 = cv2.imread(img1path,cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2path,cv2.IMREAD_UNCHANGED)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    img1resized = cv2.resize(img1, dsize=(0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_LINEAR)
    img2resized = cv2.resize(img2, dsize=(0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_LINEAR)

    desc = np.full((200, 300, 3), 255, dtype=np.uint8) # program description window
    cv2.putText(desc, "<How to use this program>", (20,40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0),thickness=1)
    cv2.putText(desc, "1. Pick 4 points on image1", (30,80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
    cv2.putText(desc, "2. Pick 4 points on image2", (30,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
    cv2.putText(desc, "You have to double click", (40,140), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
    cv2.putText(desc, "on the images!", (80,160), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))

    while True:

        cv2.imshow('Image 2', img2resized)
        cv2.imshow('Image 1', img1resized)
        cv2.imshow('How to use this application', desc)

        if i<5:
            cv2.setMouseCallback("Image 1", pickPoint, img1resized)
        else:
            cv2.setMouseCallback("Image 2", pickPoint, img2resized)

        # to check the saved images
        cv2.imshow("a1", a1)
        cv2.imshow("a2", a2)
        cv2.imshow("a3", a3)
        cv2.imshow("a4", a4)
        cv2.imshow("b1", b1)
        cv2.imshow("b2", b2)
        cv2.imshow("b3", b3)
        cv2.imshow("b4", b4)

        if cv2.waitKey(20) & 0xFF == 27: # esc key to end!
            break
    cv2.destroyAllWindows()


msg = input("To start the program, enter 'start': ")
if str(msg) == 'start':
    cornerDetect()
else:
    print("Bye!")