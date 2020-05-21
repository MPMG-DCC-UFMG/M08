# -*- coding: utf-8 -*-
import datetime
import cv2
import numpy as np
import math

# função retorna faces de uma imagem
def get_faces_mtcnn(pimg, detector, timing_detect=[] ):
    dy = int(pimg.shape[0] * 0.05)
    dx = int(pimg.shape[1] * 0.05)
    pimg = cv2.copyMakeBorder(pimg, dy, dy, dx, dx,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])

    tp1 = datetime.datetime.now()
    result = detector.detect_faces(pimg)
    tp2 = datetime.datetime.now()
    timing_detect.append((tp2-tp1).total_seconds())

    faces = []
    for r in result:
        #if r['confidence']<=confidence:
        #    continue
        bounding_box = r['box']
        kp = r['keypoints']
        # coordenadas da detecção na imagem original
        y0 = max(bounding_box[1], 0)
        y1 = min(bounding_box[1] + bounding_box[3], pimg.shape[0])
        x0 = max(bounding_box[0], 0)
        x1 = min(bounding_box[0] + bounding_box[2], pimg.shape[1])

        # aumento da área detectada
        y0n = max(y0 - int(bounding_box[3] * 0.6), 0)
        y1n = min(y1 + int(bounding_box[3] * 0.6), pimg.shape[0])
        x0n = max(x0 - int(bounding_box[2] * 0.6), 0)
        x1n = min(x1 + int(bounding_box[2] * 0.6), pimg.shape[1])

        coord = (x0-dx, y0-dy, x1-dx, y1-dy)

        img_face = pimg[y0n:y1n, x0n:x1n]

        # calcula centro da imagem detectada original - sem aumento
        cy = int((y0 + y1) / 2) - y0n
        cx = int((x0 + x1) / 2) - x0n

        # rotaciona imagem
        at = math.atan2(kp['right_eye'][1] - kp['left_eye'][1],
                        kp['right_eye'][0] - kp['left_eye'][0])
        gr = math.degrees(at)
        # print(gr)

        # cria uma borda
        bordv = int(0.15 * (y1n - y0n) / 2)
        bordh = int(0.15 * (x1n - x0n) / 2)

        imgr = cv2.copyMakeBorder(img_face, bordv, bordv, bordh, bordh, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        #plt.figure()
        #plt.imshow(imgr)
        #plt.show()
        #print(bordv, bordh)
        rm = cv2.getRotationMatrix2D((int(imgr.shape[0] / 2), int(imgr.shape[1] / 2)), gr, 1.0)

        imgr = cv2.warpAffine(imgr, rm, imgr.shape[1::-1], flags=cv2.INTER_LINEAR)
        #plt.figure()
        #plt.imshow(imgr)
        #plt.show()
        # img_face = imgr

        res2 = None
        try:
            res2 = detector.detect_faces(imgr)
        except Exception as ex:
            print("Erro em detect_faces", dy, dx, imgr.shape)
            cv2.imwrite("img_{:}_{:}_{:}_{:}.jpg".format(dy, dx, imgr.shape[0], imgr.shape[1]), imgr)
            print("Erro em detect_faces", ex)

        imgr_cy, imgr_cx = imgr.shape[0]/2, imgr.shape[1]/2

        fc2 = []
        min_dc = imgr_cy + imgr_cx
        min_img = None
        for r2 in res2:
            bb2 = r2['box']
            # coordenadas da detecção na imagem original
            y02 = max(bb2[1], 0)
            y12 = min(bb2[1] + bb2[3], imgr.shape[0])
            x02 = max(bb2[0], 0)
            x12 = min(bb2[0] + bb2[2], imgr.shape[1])
            dc = math.sqrt(((y02+y12)/2 - imgr_cy)**2 + ((x02+x12)/2 - imgr_cx)**2)

            # aumento da área detectada
            y0n2 = max(y02 - int(bb2[3] * 0.08), 0)
            y1n2 = min(y12 + int(bb2[3] * 0.08), imgr.shape[0])
            x0n2 = max(x02 - int(bb2[2] * 0.08), 0)
            x1n2 = min(x12 + int(bb2[2] * 0.08), imgr.shape[1])
            img_face = imgr[y0n2:y1n2, x0n2:x1n2]

            if dc < min_dc:
                min_dc = dc
                min_img = img_face
            #plt.figure()
            #plt.imshow(img_face)
            #plt.show()
            #faces.append((img_face, coord))

        if min_img is not None:
            #plt.figure()
            #plt.imshow(min_img)
            #plt.show()
            faces.append((min_img, coord, r['confidence']))

    return faces

