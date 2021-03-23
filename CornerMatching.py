import cv2
import numpy as np
import matplotlib.pyplot as plt

# corner index
i = 1
# corner image (5*5)
corner = list()
# patch size
patch=9
# corner-cropped images in numpy array
a1 = np.zeros((7, 7))
a2 = np.zeros((7, 7))
a3 = np.zeros((7, 7))
a4 = np.zeros((7, 7))
b1 = np.zeros((7, 7))
b2 = np.zeros((7, 7))
b3 = np.zeros((7, 7))
b4 = np.zeros((7, 7))

# 각도 단위
angleRange = 36

line_coner ={}

def pickPoint(event, x, y, flags, img):  # picking Point - when mouse clicked
    global i
    h, w = img.shape
    if event == cv2.EVENT_LBUTTONDBLCLK:
        xText, yText = textLocation(x, y, h, w)
        if i < 5:
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), color=(0, 0, 0),
                          thickness=2)  # drawing rectangle on the image
            cv2.putText(img, '1-' + str(i), (xText, yText), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)
            saveCorner(img, x, y, i, h, w)
            line_coner[i] = (x,y)
        elif i >= 5 and i < 9:
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), color=(0, 0, 0),
                          thickness=2)  # drawing rectangle on the image
            cv2.putText(img, '2-' + str(i - 4), (xText, yText), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)
            saveCorner(img, x, y, i, h, w)
            line_coner[i] = (605+x, y)
        i += 1


def textLocation(x, y, h, w):  # setting text location (to avoid overlapping)
    if x + 70 < w and y - 40 > 0:  # default: top right
        xText = x + 5
        yText = y - 5
    else:
        if x - 70 > 0 and y + 30 < h:  # bottom left
            xText = x - 70
            yText = y + 30
        else:
            if x - 70 > 0 and y - 40 > 0:  # top left
                xText = x - 70
                yText = y - 5
            else:  # bottom right
                xText = x + 5
                yText = y + 30

    return xText, yText


def saveCorner(img, x, y, i, h, w):  # save cropped corner images
    global a1, a2, a3, a4, b1, b2, b3, b4

    pointX, pointY = x, y
    if pointX < 3:
        pointX = 3
    if pointY < 3:
        pointY = 3
    if pointX + 4 > w:
        pointX -= 4
    if pointY + 4 > h:
        pointY -= 4

    if i == 1:
        a1 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(a1)
    elif i == 2:
        a2 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(a2)
    elif i == 3:
        a3 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(a3)
    elif i == 4:
        a4 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(a4)
    elif i == 5:
        b1 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(b1)
    elif i == 6:
        b2 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(b2)
    elif i == 7:
        b3 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(b3)
    elif i == 8:
        b4 = img[pointY - 3:pointY + 4, pointX - 3:pointX + 4].copy()
        corner.append(b4)


# Calculate Gradient and Magnitude and Angle
# input - edge : 책의 모서리 (patch사이즈+1)크기의 이미지 리스트
#      - PATCH : patch size (if 5*5, then 5)
# output - edgeinfo : 변화량과 각도를 담은 리스트 [a1의[mag,ori],a2의[mag,ori]...]
def get_Mag_Ori(edge, PATCH):
    cornerinfo = list()
    for e in edge:
        sobelX = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3)
        sobelY = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3)
        sobelX = sobelX[1:PATCH + 1, 1:PATCH + 1]
        sobelY = sobelY[1:PATCH + 1, 1:PATCH + 1]
        mag, ori = cv2.cartToPolar(sobelX, sobelY, angleInDegrees=True)
        cornerinfo.append([mag, ori])
    return cornerinfo;


# 히스토그램 normalization
def normalizeHisto(histo):
    total = 0.0;
    for i in range(len(histo)):
        total += histo[i] ** 2
    total = total ** 0.5
    for i in range(len(histo)):
        histo[i] = histo[i] / total;
    return histo


# 히스토그램 리스트 만들기
# input cornerinfo -> [mag, ori]
def getHisto(cornerinfo):
    histo = [0 for i in range(int(360 / angleRange))]
    mag = np.array(cornerinfo[0]).flatten().tolist()
    ori = np.array(cornerinfo[1]).flatten().tolist()
    for i in range(len(ori)):
        ratio = (angleRange - (ori[i] % 360 % angleRange)) / angleRange
        histo[int((ori[i] % 360) / angleRange)] += mag[i] * ratio
        histo[(int((ori[i] % 360) / angleRange) + 1) % 10] += mag[i] * (1 - ratio)
    return normalizeHisto(histo)


# 히스토그램 이미지 출력
def drawHisto(histo, title):
    rangeValue = []
    for i in range(int(360 / angleRange)):
        rangeValue.append(36 * i)
    x = np.arange(int(360 / angleRange))
    plt.figure(figsize=(4, 4))
    plt.bar(x, histo, width=1, align='center')
    plt.xticks(x, rangeValue)
    plt.xlim([0, 360 / angleRange])
    plt.ylim([0, 1])
    plt.title(title)


def cornerDetect():  # main function for corner detecting
    img1path = './imgs/1st.jpg'
    img2path = './imgs/2nd.jpg'


    img1 = cv2.imread(img1path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2path, cv2.IMREAD_UNCHANGED)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1resized = cv2.resize(img1, dsize=(0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_LINEAR)
    img2resized = cv2.resize(img2, dsize=(0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_LINEAR)

    h,w =img1resized.shape

    desc = np.full((200, 300, 3), 255, dtype=np.uint8)  # program description window
    cv2.putText(desc, "<How to use this program>", (20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(0, 0, 0), thickness=1)
    cv2.putText(desc, "1. Pick 4 points on image1", (30, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 0))
    cv2.putText(desc, "2. Pick 4 points on image2", (30, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 0))
    cv2.putText(desc, "You have to double click", (40, 140), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 0))
    cv2.putText(desc, "on the images!", (80, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))

    while True:

        cv2.imshow('Image 2', img2resized)
        cv2.imshow('Image 1', img1resized)
        cv2.imshow('How to use this application', desc)

        if i < 5:
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

        if i == 9:  # end to select
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(20) & 0xFF == 27:  # esc key to end!
            return
    cv2.imshow('Image 2', img2resized)
    cv2.imshow('Image 1', img1resized)
    # calculate Magnitude and Orientation
    result = get_Mag_Ori(corner, patch)
    histo = []
    for k in range(len(result)):
        histo.append(getHisto(result[k]))
        drawHisto(histo[k], str(int(k / 4 + 1)) + '-' + str(k % 4 + 1))
    plt.show()

    # 이미지 가로로 붙이기
    addh = cv2.hconcat([img1resized, img2resized])
    cv2.imshow('FINAL', addh)

    # 각각 10개의 히스토그램의 절대값 차이가 제일 적은 것을 선택
    ind = []
    for a in range(0,4):
        odd_list =[]
        for b in range(4,8):
            sum = 0
            for e in range(10):
                odd_val = abs(histo[a][e]-histo[b][e])
                sum +=odd_val
            odd_list.append(sum)
        min_index = odd_list.index(min(odd_list))
        ind.append(5+min_index)

    # 선 그리기
    black = (0,0,0)
    for s in range(4):
        cv2.line(addh,line_coner[1+s],line_coner[ind[s]],black ,4)

    cv2.imshow('FINAL',addh)
    cv2.waitKey(0)




msg = input("To start the program, enter 'start': ")
if str(msg) == 'start':
    cornerDetect()
else:
    print("Bye!")
