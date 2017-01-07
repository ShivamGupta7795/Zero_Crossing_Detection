import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal 

image = cv2.imread('HW4/UBCampus.jpg',0)
#cv2.imshow('UB IMAGE',image)

w= len(image)
h= len(image[0])

imagedog=[[0 for x in range(w)]for y in range(h)]
edges=np.zeros((w,h))
imagesobel=np.zeros((w,h))
strongedge=np.zeros((w,h))

kernel =[[0 for x in range(7)]for y in range(7)]

kernel[0][0]=0
kernel[0][1]=0
kernel[0][2]=-1
kernel[0][3]=-1
kernel[0][4]=-1
kernel[0][5]=0
kernel[0][6]=0

kernel[1][0]=0
kernel[1][1]=-2
kernel[1][2]=-3
kernel[1][3]=-3
kernel[1][4]=-3
kernel[1][5]=-2
kernel[1][6]=0

kernel[2][0]=-1
kernel[2][1]=-3
kernel[2][2]=5
kernel[2][3]=5
kernel[2][4]=5
kernel[2][5]=-3
kernel[2][6]=-1

kernel[3][0]=-1
kernel[3][1]=-3
kernel[3][2]=5
kernel[3][3]=16
kernel[3][4]=5
kernel[3][5]=-3
kernel[3][6]=-1

kernel[4][0]=-1
kernel[4][1]=-3
kernel[4][2]=5
kernel[4][3]=5
kernel[4][4]=5
kernel[4][5]=-3
kernel[4][6]=-1

kernel[5][0]=0
kernel[5][1]=-2
kernel[5][2]=-3
kernel[5][3]=-3
kernel[5][4]=-3
kernel[5][5]=-2
kernel[5][6]=0

kernel[6][0]=0
kernel[6][1]=0
kernel[6][2]=-1
kernel[6][3]=-1
kernel[6][4]=-1
kernel[6][5]=0
kernel[6][6]=0



image=  np.asarray(image)
#image = np.abs(image)
print "image",image


kernel = np.asarray(kernel)
print "kernel",kernel

imagedog = cv2.filter2D(image, -1, kernel, anchor=(-1,-1))
#imagedog=cv2.convertScaleAbs(imagedog)
#edges = cv2.Canny(imagedogabs, 50, 50)

#retval, edges= cv2.threshold(imagedogabs, 50, 255, cv2.THRESH_BINARY)

#gaus = cv2.adaptiveThreshold(imagedogabs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115,1)

print "imagedog"
imagedog = np.asarray(imagedog)
print imagedog



#print "gaus"
#gaus= np.asarray(gaus)
#print gaus



#print "imagedogabs"
#imagedogabs= np.asarray(imagedogabs)
#print imagedogabs


cv2.imshow('DoG IMAGE',imagedog)
imagedg = cv2.filter2D(image, cv2.CV_32F, kernel, anchor=(-1,-1))
#cv2.imshow('GAUS IMAGE',gaus)

#cv2.imshow('imagedogAbs IMAGE',imagedogabs)



for x in range(0,w-1):
    for y in range(0,h-1):
        if(imagedg[x][y]*imagedg[x+1][y]<0 or imagedg[x][y]*imagedg[x][y+1]<0):
            edges[x][y]=1
            
print "edges"
edges= np.asarray(edges)
print edges
cv2.imshow('Zero Crossings IMAGE',edges)

imagesobeldx=cv2.Sobel(image, cv2.CV_16S, 1, 0, 3, 1)
imagesobeldy=cv2.Sobel(image, cv2.CV_16S, 0, 1, 3, 1)

imagesobel = np.hypot(imagesobeldx, imagesobeldy)

imagesobel = np.asarray(imagesobel)
imagesobel = np.abs(imagesobel)
imagesobel = np.uint8(imagesobel)
print "imagesobel",imagesobel

#cv2.imshow("imagesobel",imagesobel)

#strongedge = np.logical_and(edges, imagesobel)
#
for x in range(0,w):
    for y in range(0,h):
        if(imagesobel[x][y]>30):
            strongedge[x][y]=1
        else:
            strongedge[x][y]=0
            
strongedge= np.asarray(strongedge)
print "strongedge",strongedge

#plt.imshow(strongedge,cmap='gray',interpolation='bicubic')
#plt.show()
cv2.imshow("Strong Edges", strongedge)
cv2.imwrite("Strongedgeimg.png",strongedge)

lpkernel = kernel =[[0 for x in range(5)]for y in range(5)]

lpkernel[0][0] = 0
lpkernel[0][1] = 0
lpkernel[0][2] = 1
lpkernel[0][3] = 0
lpkernel[0][4] = 0

lpkernel[1][0] = 0
lpkernel[1][1] = 1
lpkernel[1][2] = 2
lpkernel[1][3] = 1
lpkernel[1][4] = 0


lpkernel[2][0] = 1
lpkernel[2][1] = 2
lpkernel[2][2] = -16
lpkernel[2][3] = 2
lpkernel[2][4] = 1

lpkernel[3][0] = 0
lpkernel[3][1] = 1
lpkernel[3][2] = 2
lpkernel[3][3] = 1
lpkernel[3][4] = 0

lpkernel[4][0] = 0
lpkernel[4][1] = 0
lpkernel[4][2] = 1
lpkernel[4][3] = 0
lpkernel[4][4] = 0


lpkernel = np.asarray(lpkernel)
imagelp = [[0 for x in range(w)]for y in range(h)]


imagelp = cv2.filter2D(image,cv2.CV_32F, lpkernel)
imagel = cv2.filter2D(image,-1, lpkernel)


imagel= np.asarray(imagel)

print "imagel",imagel

cv2.imshow('LoG IMAGE',imagel)

logzero = np.zeros((w,h))

for x in range(0,w-1):
    for y in range(0,h-1):
        if((imagelp[x][y]*imagelp[x+1][y])<0 or (imagelp[x][y]*imagelp[x][y+1])<0):
            logzero[x][y]=1

logzero = np.asarray(logzero)
print "logzero",logzero

cv2.imshow("logzero",logzero)

differimg = np.zeros((w,h))

for x in range(0,w):
    for y in range(0,h):
        differimg[x][y]=imagelp[x][y]-strongedge[x][y]
    
differimg = np.asarray(differimg)
print differimg
cv2.imshow("differimg",differimg)

cv2.imwrite("grayimg.png",image)
cv2.imwrite("DoGimg.png",imagedog)
cv2.imwrite("ZeroCrossimg.png",edges)
cv2.imwrite("LoGimg.png",imagelp)
       
        









 

 


