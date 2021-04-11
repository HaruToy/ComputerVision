import cv2
import numpy as np

i = 0
pts_src = np.array([0, 0])
pts_dst = np.array([0, 0])


def savePoint(event, x, y, flags, img):
    global i, pts_src, pts_dst
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(i)
        print(pts_src)
        print(pts_dst)
        if i == 0:
            pts_src = np.array([x, y])
            cv2.circle(img, (x, y), radius=6, color=(0, 0, 0), thickness=2)
        if i > 0 and i < 4:
            pts_src = np.vstack((pts_src, [x, y]))
            cv2.circle(img, (x, y), radius=6, color=(0, 0, 0), thickness=2)
        if i == 4:
            pts_dst = np.array([x, y])
            cv2.circle(img, (x, y), radius=6, color=(0, 0, 0), thickness=2)
        if i > 4 and i < 8:
            pts_dst = np.vstack((pts_dst, [x, y]))
            cv2.circle(img, (x, y), radius=6, color=(0, 0, 0), thickness=2)

        i += 1
def makeHomography(p_src,p_dst):

    x1 = p_src[0][0]
    y1 = p_src[0][1]
    x2 = p_src[1][0]
    y2 = p_src[1][1]
    x3 = p_src[2][0]
    y3 = p_src[2][1]
    x4 = p_src[3][0]
    y4 = p_src[3][1]

    xp1 = p_dst[0][0]
    yp1 = p_dst[0][1]
    xp2 = p_dst[1][0]
    yp2 = p_dst[1][1]
    xp3 = p_dst[2][0]
    yp3 = p_dst[2][1]
    xp4 = p_dst[3][0]
    yp4 = p_dst[3][1]

    A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
                  [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
                  [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
                  [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
                  [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],
                  [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
                  [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
                  [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]
                  ])

    aa = A.astype(np.float64)

    U, S, VT =cv2.SVDecomp(aa, flags= cv2.SVD_FULL_UV)
    V = np.transpose(VT)
    b = np.array(V).T[8]
    H = np.array([[b[0], b[1], b[2]],
                  [b[3], b[4], b[5]],
                  [b[6], b[7], b[8]]])
    print(H)
    return H



def InverseMapping(h, im_src, im_dst, im_src_mask):
    h = np.linalg.inv(h)
    img_dst_point = []
    for j in range(0, im_dst.shape[1]):
        for i in range(0, im_dst.shape[0]):
            point = [j, i, 1]
            img_dst_point.append(point)
    img_dst_point = np.array(img_dst_point).transpose()

    img_src_point = np.matmul(h, img_dst_point)
    third = img_src_point[2:]
    img_src_point = img_src_point[:2] / third
    im_out = im_dst

    img_t = np.floor(img_src_point).astype(int)
    a = np.subtract(img_src_point, img_t)

    for i in range(img_src_point.shape[1]):
        x_d = img_t[0][i]
        y_d = img_t[1][i]
        alpha = a[0][i]
        beta = a[1][i]
        if x_d > 0 and x_d < im_src.shape[1] and y_d > 0 and y_d < im_src.shape[0] and im_src_mask[y_d, x_d][0] != 0 and \
                im_src_mask[y_d, x_d][1] != 0 and im_src_mask[y_d, x_d][2] != 0:
            im_out[img_dst_point[1][i], img_dst_point[0][i]] = np.round(
                (1 - alpha) * (1 - beta) * im_src[y_d, x_d] + alpha * (1 - beta) * im_src[y_d, x_d + 1] + alpha * beta *
                im_src[y_d + 1, x_d + 1] + (1 - alpha) * beta * im_src[y_d + 1, x_d], 0).astype(int)
    return im_out


if __name__ == '__main__':

    imgpath = './gogh.jpg'
    targetpath = './target.jpeg'

    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    target = cv2.imread(targetpath, cv2.IMREAD_UNCHANGED)

    im_src = cv2.resize(img, dsize=(0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    im_dst = cv2.resize(target, dsize=(0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

    while True:

        cv2.imshow('Target', im_dst)
        cv2.imshow('Image', im_src)

        if i < 4:
            cv2.setMouseCallback("Image", savePoint, im_src)
        else:
            cv2.setMouseCallback("Target", savePoint, im_dst)

        if i == 8:  # end to select
            cv2.destroyAllWindows()
            print('All 8 points selected!')
            break
        if cv2.waitKey(20) & 0xFF == 27:  # esc key to end!
            print("Program End!")
            break

    if len(pts_src) == 4 and len(pts_dst) == 4:
        black = np.full((im_src.shape[0], im_src.shape[1]), 0, dtype=np.uint8)
        im_out = cv2.fillConvexPoly(black, pts_src, (255, 255, 255))
        im_src_mask = cv2.bitwise_and(im_src, im_src, mask=im_out)

        # Calculate Homography
        H = makeHomography(pts_src,pts_dst)

        # Inverse Mapping
        im_out = InverseMapping(H, im_src, im_dst, im_src_mask)

        # Warp source image to destination based on homography
        # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

        # Display images
        cv2.imshow("Warped Source Image", im_out)
        cv2.imshow("Warped Source1 Image", im_src)

        cv2.waitKey(0)
