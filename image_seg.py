import numpy as np
import cv2
from matplotlib import pyplot
from scipy import signal
import math

image= cv2.imread('HW4/MixedVegetables.jpg',0)


w= len(image)
h= len(image[0])

testimage=np.zeros((2*w,2*h))

image = np.asarray(image)
image = np.abs(image)

x,y,m,n=0,0,0,0
while x<w and m<2*w:
    while y<h and n<2*h:
        testimage[m][n]= image[x][y]
        y+=1
        n+=2
    x+=1
    m+=2    
    y,n=0,0
        
        

x,y,m,n=0,0,0,1
while x<w and m<2*w:
    while y<h-1 and n<2*h:
        testimage[m][n] = math.fabs(image[x][y]-image[x][y+1])
        y+=1
        n+=2
    x+=1
    m+=2
    y,n=0,1

x,y,m,n=0,0,1,0
while y<h and n<2*h:
    while x<w-1 and m<2*w:
        testimage[m][n] = math.fabs(image[x][y]-image[x+1][y])
        x+=1
        m+=2
    y+=1
    n+=2
    x,m=0,1

testimage= np.asarray(testimage)
print "testimage0",testimage 


m,n,i=0,0,0
for m in range(0,2*w):
    for n in range(0,2*h):
        if(testimage[m][n]<30):
            testimage[m][n]=0
            i+=1
testimage= np.asarray(testimage)
print "testimage1",testimage  
cv2.imshow('testimage1',testimage)
          
        

j=0
for x in range(0,2*w-1):
    for y in range(0,2*h-1):
        if(math.fabs(testimage[x][y]-testimage[x+1][y])>50 or math.fabs(testimage[x][y]-testimage[x][y+1])>50):
            testimage[x][y]=255
            j+=1


print "j=",j
testimage= np.asarray(testimage)
print testimage
cv2.imshow("testimage2",testimage)
pyplot.imshow(testimage, cmap='gray')
pyplot.show()          
        

