import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX
#Read the image 
img=cv2.imread('pipes.jpg')
#cv2.imshow('pipes',img)
img=cv2.medianBlur(img,3)

#Image filtering
blur_hor = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((11,1,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
blur_vert = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((1,11,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
mask = ((img[:,:,0]>blur_hor*1.2) | (img[:,:,0]>blur_vert*1.2)).astype(np.uint8)*255

#cv2.imshow('mask',mask)

#Detect circle using cv2.HoughCircles.The parameters depend on the input. 
circles=cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,dp=1,minDist=60,param1=150,param2=25, minRadius=0,maxRadius=50)
circles = np.uint16(np.around(circles))

#Get the number of circles detected
number_of_circles=circles.shape[1]
#print("Number of pipes detected: ",number_of_circles)

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,215,0),2)
#Display the text    
cv2.putText(img,"Total Detected pipes: "+'{}'.format(number_of_circles),(100,290),font,1,(255,255,255),2)

cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
