# Gesture-boot
Using unique human gesture detection to automatically turn on the phone, similar to the face recognition function
size is the resolution of the whole image, (x, y) is the normalized coordinates of the center of the target object for the whole image, (w, h) is the normalized width and height of the bounding box of the target object for the whole image.
dw = 1./(size[0])  
dh = 1./(size[1])  
x = (box[0] + box[1])/2.0 - 1  
y = (box[2] + box[3])/2.0 - 1  
w = box[1] - box[0]  
h = box[3] - box[2]  
x = x*dw  
w = w*dw  
y = y*dh  
h = h*dh  
