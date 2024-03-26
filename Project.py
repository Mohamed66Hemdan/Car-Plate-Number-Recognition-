# Library 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import imutils  
import easyocr 

# OCR Architecture Code
# 1) - Read Image
img = cv2.imread("WhatsApp Image 2023-05-09 at 11.26.31 AM.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB));

# 2) - Preprocessing [Noise Reduction & Edge Eetection ]
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
edged = cv2.Canny(bfilter, 30, 200) 
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB));

# 3) - Draw Contours
# Specify The Edges In Image 
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# [Output Shape ]
contourss = imutils.grab_contours(keypoints) # [Output Shape ]
# Max Counters ['Area بيتحدد عن طريق ال '] 
contours = sorted(contourss, key=cv2.contourArea, reverse=True)[:10] # reverse ['برتبهم']
len(contours)

# Specify the Plate Number 
location = None
# Geometric Shape Detection # [countour , distance [ممكن اعتبرها من ضمن المستطيل بتاعي ] , closed]
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
len(location)    
# location
# approx[1][0][0]

# mask Window Empty [Black]
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], -1 ,255, -1)
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB));

new_image = cv2.bitwise_and(img, img, mask=mask)
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)); # [بيبدل بين الصوره ونفسها علي الماسك]

(x,y) = np.where(mask == 255) ## [search of the value with pixel 255] [specify the cooridnate]
# to specify The rectangle [top left , bottom right]
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB));

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
result

text = result[0][-2]
# b,g,r,a = 0,255,0,0
# # font = cv2.FONT_HERSHEY_SIMPLEX
# fontpath = "arial.ttf" # <== download font
# font = ImageFont.truetype(fontpath,32)
# img_pil = Image.fromarray(img)
# draw = ImageDraw.Draw(img_pil)
# draw.text((380, 200),  text, font = font,fill = (b, g, r, a))
# img = np.array(img_pil)
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB));

# ! pip install arabic-reshaper
import cv2 
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
from PIL import ImageFont, ImageDraw, Image
# fontpath = "arial.ttf" # <== download font
# font = ImageFont.truetype(fontpath, 32)
# img_pil = Image.fromarray(img)
# draw = ImageDraw.Draw(img_pil)
# draw.text((0, 80),'محمد احمد', font = font)
# img = np.array(img_pil)
# # cv2.imshow('window_name',img) 
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
