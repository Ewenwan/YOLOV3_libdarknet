import sys, os ,cv2
import numpy as np
sys.path.append(os.path.join(os.getcwd(),'detector2/'))

import darknet as dn

def process_d2box(b):
	mess = b[0]
	mess = bytes.decode(mess)# bytes to string
	confidence = b[1]
	box = b[2]
	x = box[0]
	y = box[1]
	w = box[2]
	h = box[3]
	left = float(x)-0.5*float(w)
	top = float(y)-0.5*float(h)
	right = float(x)+0.5*float(w)
	bot = float(y) + 0.5*float(h)
	return (left,top,right,bot,mess,confidence)#(left,top,right,bot)

# Darknet
####################### for python3 ##########################
net = dn.load_net("detector2/yolov3.cfg".encode("ascii"), "detector2/yolov3.weights".encode("ascii"), 0)
meta = dn.load_meta("detector2/coco.data".encode("ascii"))

cv2.namedWindow('YOLOV3')
img = cv2.imread('dog.jpg')
d2_boxes = dn.detect_np(net, meta, img, 0.55, 0.55, 0.45)# net, meta, image, thresh, hier_thresh, nms
for b in d2_boxes:
	boxResults = process_d2box(b)
	if boxResults is None:
		continue
	left, top, right, bot, mess, confidence = boxResults
	cv2.rectangle(img, (int(left), int(top)), (int(right), int(bot)),
					  (255, 0, 0), 2)
	cv2.putText(img, str(mess) + ":" + str('%.1f' % (float(confidence) * 100)) + "%", (int(left), int(top) - 8),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # hy change:2017/12/18
cv2.imshow('YOLOV3',img)
cv2.waitKey()
cv2.destroyAllWindows()






