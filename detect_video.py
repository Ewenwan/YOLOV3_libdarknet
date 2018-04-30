import sys, os ,cv2,re
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

# OpenCV

video = 'PETS09-S2L1.avi'
cap = cv2.VideoCapture(video)
#cap = cv2.VideoCapture(0)

cv2.namedWindow('YOLOV3')
sucess,frame = cap.read()

f_name = re.split('[/]',video)# changed by hy 2018/4/26 extract file name
out_name = f_name[-1]
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(
            'output_{}'.format(out_name), fourcc, 25, (width, height))

while sucess and cv2.waitKey(1) == -1:

	d2_boxes = dn.detect_np(net, meta, frame, 0.55, 0.55, 0.45)# net, meta, image, thresh, hier_thresh, nms
	detections = []
	scores = []
	for b in d2_boxes:
		boxResults = process_d2box(b)
		if boxResults is None:
			continue
		left, top, right, bot, mess, confidence = boxResults
		detections.append(np.array([left, top, right, bot]).astype(np.float64))
		scores.append(confidence)
		cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bot)),
					  (255, 0, 0), 2)
		cv2.putText(frame, str(mess) + ":" + str('%.1f' % (float(confidence) * 100)) + "%", (int(left), int(top) - 8),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # hy change:2017/12/18
	result = frame
	cv2.imshow('YOLOV3',result)
	videoWriter.write(result)
	sucess,frame = cap.read()
cv2.destroyAllWindows()
cap.release()
videoWriter.release()






