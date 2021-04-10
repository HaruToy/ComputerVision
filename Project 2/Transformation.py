import cv2
import numpy as np

def InverseMapping(h,im_src,im_dst,im_src_mask):
  h = np.linalg.inv(h)
  img_dst_point=[]
  for j in range(0,im_dst.shape[1]):
    for i in range(0,im_dst.shape[0]):
      point=[j,i,1]
      img_dst_point.append(point)
  img_dst_point=np.array(img_dst_point).transpose()

  img_src_point=np.matmul(h,img_dst_point)
  third=img_src_point[2:]
  img_src_point=img_src_point[:2]/third
  img_src_point=np.round(img_src_point, 0).astype(np.int)
  im_out=im_dst

  for i in range(img_src_point.shape[1]):
      x_d=  img_src_point[0][i]
      y_d = img_src_point[1][i]
      if x_d>0 and x_d<im_src.shape[1] and y_d>0 and y_d<im_src.shape[0] and im_src_mask[y_d,x_d][0]!=0 and im_src_mask[y_d,x_d][1]!=0 and im_src_mask[y_d,x_d][2]!=0:
        im_out[img_dst_point[1][i],img_dst_point[0][i]] = im_src[y_d,x_d]
  return im_out

if __name__ == '__main__':

  # Read source image.
  im_src = cv2.imread('book2.jpg')
  # our corners of the book in source image
  pts_src = np.array([[252, 207], [424, 298], [252, 533], [56, 379]])


  # Read destination image.
  im_dst = cv2.imread('book1.jpg')
  cv2.imshow("Warped Source 2Image", im_dst)

  # Four corners of the book in destination image.
  pts_dst = np.array([[114, 110], [381, 130], [392, 502], [47, 478]])

  black = np.full((im_src.shape[0],im_src.shape[1]),0,dtype=np.uint8)
  im_out = cv2.fillConvexPoly(black, pts_src, (255,255,255))
  im_src_mask=cv2.bitwise_and(im_src,im_src,mask = im_out)

  # Calculate Homography
  h, status = cv2.findHomography(pts_src, pts_dst)

  # Inverse Mapping
  im_out=InverseMapping(h,im_src,im_dst,im_src_mask)

  # Warp source image to destination based on homography
  #im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

  # Display images
  cv2.imshow("Warped Source Image",im_out)
  cv2.imshow("Warped Source1 Image", im_src)

  cv2.waitKey(0)


