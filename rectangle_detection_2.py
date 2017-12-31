import cv2 as cv      #import opencv
import numpy as np    #import numpy

import math


green = (0, 255, 0)


#function to show image 
def show_image(image):
    
    cv.imshow('frame',image)
    if cv.waitKey(10000) & 0xFF == ord('q'):
        return
    cv.destroyAllWindows()
        

def overlay_mask(mask, image):
	
    rgb_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    img = cv.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img



#finding contour of rectangle inside big rectangle
def find_inside_contour(image):
    image_new=image.copy()
    ret,thresh=cv.threshold(image,127,255,0)
    image,contours,hierarchy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    
    contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
    inside_contour = max(contour_sizes, key=lambda x: x[0])[1]
    
    mask = np.zeros(image.shape, np.uint8)
    cv.drawContours(mask, [inside_contour], -1, 255, -1)
    return inside_contour, mask


def find_biggest_contour(image4):
    
     image=image4.copy()                #copy name into image
     print (image4.shape)               #print image shape which is in numpy array
     
     
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
    #show_image(image)
    
    image_blur = cv.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv.cvtColor(image_blur, cv.COLOR_RGB2HSV)

    show_image(image_blur_hsv)
    min_white = np.array([0, 0,235 ])
    max_white = np.array([5, 5, 248])
    #layer
    mask1 = cv.inRange(image_blur_hsv, min_white, max_white)
    
    show_image(mask1)
    cv.imwrite("data/First_mask_2.jpg",mask1)

    min_white2 = np.array([0, 0, 240])
    max_white2 = np.array([50, 10, 256])
    
    mask2 = cv.inRange(image_blur_hsv, min_white2, max_white2)
    show_image(mask2)
    cv.imwrite("data/Second_mask_2.jpg",mask2)
    mask = mask1 + mask2
    show_image(mask)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
 
    mask_clean = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel)


    big_rectangle_contour, mask_rectangles = find_biggest_contour(mask_clean)

    inside_rectangle_contour,mask_rectangles1=find_inside_contour(mask_clean)
    overlay = overlay_mask(mask_clean, image)


    rectangled = rectangle_contour(overlay, big_rectangle_contour)
    show_image(rectangled)
    
   
    bgr = cv.cvtColor(rectangled, cv.COLOR_RGB2BGR)
    
    return bgr





image=cv.imread('data/rectangle.jpg')
image4=image.copy()

result = find_rectangle(image)
cv.imwrite('data/final_output_2.jpg', result)
