import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
import math


#mage=cv.VideoCapture(0)
green = (0, 255, 0)

def show_image(image):
    
    cv.imshow('frame',image)
    if cv.waitKey(10000) & 0xFF == ord('q'):
        return
        
        
    cv.destroyAllWindows()
        

def overlay_mask(mask, image):
	
    rgb_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    img = cv.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img







def find_biggest_contour(image4):
    #image=image.copy
     image=image4.copy()
     print (image4.shape)
     #imgray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
     
     ret,thresh = cv.threshold(image4,127,255,0)
     #show_image(thresh)
     image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
     contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
     biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

     mask = np.zeros(image.shape, np.uint8)
     cv.drawContours(mask, [biggest_contour], -1, 255, -1)
     return biggest_contour, mask

    
    
    
    
    
 
    
    
    
    
    
    



def rectangle_contour(image, contour):
    # Bounding ellipse
    image_with_rectangle=image.copy()
    rectangle= cv.minAreaRect(contour)
    box = cv.boxPoints(rectangle)
    box = np.int0(box)
    image_with_rectangle = cv.drawContours(image_with_rectangle,[box],0,(0,0,255),2)
    

    return image_with_rectangle







    
    
def find_rectangle(image):
 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


    max_dimension = max(image.shape)
   
    scale = 700/max_dimension
  
    image = cv.resize(image, None, fx=scale, fy=scale)
    
    
    image_blur = cv.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv.cvtColor(image_blur, cv.COLOR_RGB2HSV)


    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv.inRange(image_blur_hsv, min_red, max_red)
    cv.imwrite("image1.jpg",mask1)

    show_image(mask1)
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv.inRange(image_blur_hsv, min_red2, max_red2)
    show_image(mask2)
    cv.imwrite("image12.jpg",mask2)
    mask = mask1 + mask2

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
 
    mask_clean = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel)


    big_rectangle_contour, mask_strawberries = find_biggest_contour(mask_clean)


    overlay = overlay_mask(mask_clean, image)


    rectangled = rectangle_contour(overlay, big_rectangle_contour)
    show_image(rectangled)
    
   
    bgr = cv.cvtColor(rectangled, cv.COLOR_RGB2BGR)
    
    return bgr
image=cv.imread('rectangle.jpg')
image4=image.copy()
print (image4.shape)
#detect it
result = find_rectangle(image)
#write the new image
cv.imwrite('yo2.jpg', result)
                
