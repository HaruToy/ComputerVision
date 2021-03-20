import cv2

#Calculate Gradient and Magnitude and Angle
#input - edge : 책의 모서리 (patch사이즈+1)크기의 이미지 리스트
#      - PATCH : patch size (if 5*5, then 5)
#output - edgeinfo : 변화량과 각도를 담은 리스트 [[mag,ori],...]
def Get_Mag_Ori(edge,PATCH):
  edgeinfo=list()
  for e in edge:
    sobelX = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3)
    sobelX=sobelX[1:PATCH+1,1:PATCH+1]
    sobelY = sobelY[1:PATCH + 1, 1:PATCH + 1]
    mag, ori = cv2.cartToPolar(sobelX, sobelY, angleInDegrees=True)
    edgeinfo.append([mag,ori])
  return edgeinfo;

#마우스 클릭시 좌표 출력
def on_mouse(event, x, y):
  if event == cv2.EVENT_LBUTTONDOWN:
    center.append([x, y])
    print(center)

#마우스 클릭시 좌표 리스트
center=list()

img=cv2.imread('1st.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,dsize=(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_LINEAR)

cv2.namedWindow('img')
cv2.setMouseCallback('img', on_mouse)
cv2.imshow("img", img)
cv2.waitKey(0)
edge=list()

PATCH=9
p=PATCH/2;

for c in center:
  img = cv2.rectangle(img, (c[0]-4,c[1]-4), (c[0]+5,c[1]+5), (255, 0, 0), 2)
  edge.append(img[c[1]-(p+1):c[1]+(p+2),c[0]-(p+1):c[0]+(p+2)])
  cv2.imshow("img", img)
cv2.waitKey(0)

print(Get_Mag_Ori(edge,PATCH))


