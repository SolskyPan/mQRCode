import numpy as np
from PIL import Image
import cv2
import os
import math
import copy

np.set_printoptions(threshold=np.inf)

##############
# open image from folder
##############
def openimage(path):
	files=os.listdir(path)
	img_list = {}
	for pic in files:
		if pic[-3:] == 'png':
			image=cv2.imread(path+pic)
			#image = cv2.blur(image,(8,8))
			img_list[pic] = image
			#cv2.imshow('ImageWindow', image)
			#cv2.waitKey()

	return img_list


##############
# check area of contour and child contours
##############
def check_contours_area_1(contour1, contour2):
	contour1_area = cv2.contourArea(contour1)
	contour2_area = cv2.contourArea(contour2)
	if contour2_area == 0:
		return False
	area_ratio = contour1_area / contour2_area
	if area_ratio > 1.0:
		return True #external
	return False

def check_contours_area_2(contour1, contour2):
	contour1_area = cv2.contourArea(contour1)
	contour2_area = cv2.contourArea(contour2)
	if contour2_area == 0:
		return False
	area_ratio = contour1_area / contour2_area
	#print('area ratio 2 = ', area_ratio)
	if area_ratio > 1.0:
		return True #internal
	return False

##################
# find center of contour
##################
def compute_center(contour):

	M=cv2.moments(contour)
	x = int(M['m10'] / M['m00'])
	y = int(M['m01'] / M['m00'])
	return x,y


def check_contours_center(x0,y0,x1,y1,x2,y2):
	distance0 = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
	distance1 = math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)
	distance2 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
	#print('distance0 =', distance0)
	#print('distance1 =', distance1)
	#print('distance2 =', distance2)
	if (distance0 + distance1 + distance2)/3 < 5:
	#if (distance0 < 5):
		return True
	return False



def proc_img(img):
	#_,gray=cv2.threshold(img,0,260,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)  #convert to binary image
	image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	image = cv2.blur(image,(10,10))
	#_,gray=cv2.threshold(image,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
	gray = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	contours,hierachy=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	#cv2.drawContours(gray,contours,-1,(128,255,0),3)
	#cv2.imshow("Keypoints", gray)
	#cv2.waitKey(1)
	hierachy = hierachy[0]
	#print(hierachy)
	centers = []
	res = np.array([])
	test = copy.deepcopy(img)
	for i in range(len(hierachy)):
		#cv2.drawContours(gray,contours[i],-1,(128,255,0),3)
		#cv2.imshow("Keypoints", gray)
		#cv2.waitKey(10)
		child = hierachy[i][2]
		#print('child =', child)
		#if child != -1:
			#cv2.drawContours(gray,contours[i],-1,(128,255,0),3)
			#cv2.imshow("Keypoints", gray)
			#cv2.destroyAllWindows()
			#cv2.waitKey(10)
			#print('child =', child)
			#print('grandchild =', hierachy[child][2])
		if child != -1 and hierachy[child][2] != -1 and hierachy[hierachy[child][2]][2] != -1:
			#print('Is there any?')
			grandchild = hierachy[child][2]
			if check_contours_area_1(contours[i], contours[child]) == True and check_contours_area_2(contours[child], contours[grandchild]) == True:
				#print("Checked?")
				x_self,y_self = compute_center(contours[i])
				x_child,y_child = compute_center(contours[child])
				x_grandchild,y_grandchild = compute_center(contours[grandchild])
				if check_contours_center(x_self,y_self,x_child,y_child,x_grandchild,y_grandchild) == True:
					centers.append([x_self,y_self,i])
					test[y_self,x_self] = [0,0,255]
	print('centers =', centers)

	#cv2.imshow("Keypoints",test)
	#cv2.waitKey(0)

	if len(centers)<3:
		return res
	
	max_distance = 0
	for i in range(len(centers)):
		for j in range(i+1, len(centers)):
			for k in range(j+1, len(centers)):
				distance0 = math.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
				distance1 = math.sqrt((centers[i][0] - centers[k][0]) ** 2 + (centers[i][1] - centers[k][1]) ** 2)
				distance2 = math.sqrt((centers[j][0] - centers[k][0]) ** 2 + (centers[j][1] - centers[k][1]) ** 2)
				if abs(distance0 - distance1) < 5:
					if abs(distance0**2 + distance1**2 - distance2**2) < (distance0**2 + distance1**2)/100:
						if distance0 >= max_distance:
							max_distance = distance0
							res = np.concatenate((contours[centers[i][2]], contours[centers[j][2]], contours[centers[k][2]]))
				elif abs(distance0 - distance2) < 5:
					if abs(distance0**2 + distance2**2 - distance1**2) < (distance0**2 + distance1**2)/100:
						if distance0 >= max_distance:
							max_distance = distance0
							res = np.concatenate((contours[centers[i][2]], contours[centers[j][2]], contours[centers[k][2]]))
				elif abs(distance1 - distance2) < 5:
					if abs(distance1**2 + distance2**2 - distance0**2) < (distance0**2 + distance1**2)/100:
						if distance1 >= max_distance:
							max_distance = distance1
							res = np.concatenate((contours[centers[i][2]], contours[centers[j][2]], contours[centers[k][2]]))
	return res

def get_qrcode_and_type(img):
	contours = proc_img(img)
	result = np.array([])
	if contours.size != 0:
		rect = cv2.minAreaRect(contours)
		#contour_areas = []
		#for contour in contours:
		#	contour_areas.append(cv2.contourArea(contour))

		#contour_area = max(contour_areas)
		#total_area = (np.max(box[:,1]) - np.max(box[:,1])) * (np.max(box[:,0]) - np.min(box[:,0]))
		
		#area_ratio = contour_area - total_area

		box = cv2.boxPoints(rect)
		box = np.int0(box)
		#print('box = ', box)
		result=copy.deepcopy(img)
		result = img[np.min(box[:,1]):np.max(box[:,1]), np.min(box[:,0]):np.max(box[:,0])]
		#cv2.drawContours(result, [box], 0, (255, 128, 255), 2)
		#cv2.drawContours(img,[contours[0]],0,(0, 125, 255),2)
		#cv2.drawContours(img,[contours[1]],0,(0, 120, 255),2)
		#cv2.drawContours(img,[contours[2]],0,(0, 120, 255),2)
		#print(contours[0])
		#cv2.imshow('img',result)
		#cv2.waitKey(1)
		#cv2.imshow('img',result)
		#cv2.waitKey(1)
		#cv2.imwrite('./saved/'+str(1)+'.png',result)

	#if result.size != 0:
	#	gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
	#	_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
	#	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#	cnt = contours[0]
	#	x,y,w,h = cv2.boundingRect(cnt)
	#	result = result[y:y+h,x:x+w]
	return result




def label_pixel(bw_img, row, col):
	res = set()
	l = []
	l.append((row,col))
	#print(row,col)
	while len(l) != 0:
		#print(res)
		pixel = l.pop()
		if (pixel in res) == False:
			res.add(pixel)

		i = pixel[0]
		j = pixel[1]

		if i>0 and bw_img[i-1][j] != 127 and ((i-1, j) in res) == False:
			l.append((i-1, j))

		if i<28 and bw_img[i+1][j] != 127 and ((i+1, j) in res) == False:
			l.append((i+1, j))

		if j>0 and bw_img[i][j-1] != 127 and ((i, j-1) in res) == False:
			l.append((i, j-1))

		if j<28 and bw_img[i][j+1] != 127 and ((i, j+1) in res) == False:
			l.append((i, j+1))

		if i>0 and j>0 and bw_img[i-1][j-1] != 127 and ((i-1, j-1) in res) == False:
			l.append((i-1, j-1))

		if i<28 and j<28 and bw_img[i+1][j+1] != 127 and ((i+1, j+1) in res) == False:
			l.append((i+1, j+1))

		if i>0 and j<28 and bw_img[i-1][j+1] != 127 and ((i-1, j+1) in res) == False:
			l.append((i-1, j+1))

		if i<28 and j>0 and bw_img[i+1][j-1] != 127 and ((i+1, j-1) in res) == False:
			l.append((i+1, j-1))


	return res





def decode_bw_list(black_white_list):
	res = np.ones([29, 29]) * 127
	overall_labels_map = {}
	overall_labels = np.zeros([29,29])
	max_overall_label = 1
	
	res[0:7, 0:7] = 255
	res[0:7, -7:] = 255
	res[-7:, 0:7] = 255

	res[0:7, 0] = 0
	res[0:7, 6] = 0
	res[0:7, -1] = 0
	res[0:7, -7] = 0

	res[0, 0:7] = 0
	res[0, -7:] = 0
	res[6, 0:7] = 0
	res[6, -7:] = 0

	res[-7:, 0] = 0
	res[-7:, 6] = 0
	res[-7, 0:7] = 0
	res[-1, 0:7] = 0

	res[2:5, 2:5] = 0
	res[2:5, -5:-2] = 0
	res[-5:-2, 2:5] = 0
	
	for bw_img in black_white_list:

		labels=np.zeros([29,29])
		label_pixel_map = {}
		max_index = 1
		for i in range(29):   # Assign labels
			for j in range(29):
				if bw_img[i][j] != 127 and labels[i][j] == 0:
					same_label_set = label_pixel(bw_img,i,j)
					label_pixel_map[max_index] = copy.deepcopy(same_label_set)
					for point in same_label_set:
						labels[point[0]][point[1]] = max_index
					max_index+=1


		for incoming_label in label_pixel_map:
			union_set = label_pixel_map[incoming_label]
			for pixel in union_set:
				if res[pixel[0]][pixel[1]] == 127:
					res[pixel[0]][pixel[1]] = bw_img[pixel[0]][pixel[1]]
			tmp_label_set = set()
			for current_label in overall_labels_map:
				if union_set.isdisjoint(overall_labels_map[current_label]) == False:
					intersection = union_set.intersection(overall_labels_map[current_label])
					point = intersection.pop()
					
					if bw_img[point[0]][point[1]] != res[point[0]][point[1]]:
						#print(point)
						for pixel in overall_labels_map[current_label]:
							#print(pixel)
							#print(res[pixel[0]][pixel[1]])
							res[pixel[0]][pixel[1]] = abs(res[pixel[0]][pixel[1]] - 255)
							#print(res[pixel[0]][pixel[1]])

					tmp_label_set.add(current_label)
					union_set = union_set.union(copy.deepcopy(overall_labels_map[current_label]))

			if len(tmp_label_set) != 0:
				for label in tmp_label_set:
					overall_labels_map.pop(label)

			if incoming_label not in overall_labels_map:
				overall_labels_map[incoming_label] = union_set
			else:
				if overall_labels_map[incoming_label].issubset(union_set):
					pass
				else:
					max_overall_label = 0
					for label in overall_labels_map:
						if label > max_overall_label:
							max_overall_label = label
					max_overall_label += 1

					overall_labels_map[max_overall_label] = overall_labels_map.pop(incoming_label)
					overall_labels_map[incoming_label] = copy.deepcopy(union_set)

		
		'''
		for label in label_pixel_map:
			for pixel in label_pixel_map[label]:
				row = pixel[0]
				col = pixel[1]

				if res[row][col] != 127:    # Already classified


					if res[row][col] != bw_img[row][col]:   # Phase inversion occurred
						for point in overall_labels_map[overall_labels[row][col]]:
							res[point[0]][point[1]] = abs(res[point[0]][point[1]]-255)

					if overall_labels[row][col] == label:
						pass
					else:
						if (label in overall_labels) == False:
							overall_labels_map[label] = overall_labels_map.pop(overall_labels[row][col])
							#overall_labels_map[label] = overall_labels_map[label].union(label_pixel_map[label])
							for point in overall_labels_map[label]:
								overall_labels[point[0]][point[1]] = label
						else:
							old_label = overall_labels[row][col]
							replace = overall_labels_map.pop(label)
							new = overall_labels_map.pop(old_label)
							overall_labels_map[old_label] = replace
							overall_labels_map[label] = new

							for point in overall_labels_map[old_label]:
								overall_labels[point[0]][point[1]] = old_label

							for point in overall_labels_map[label]:
								overall_labels[point[0]][point[1]] = label
							#overall_labels_map[label] = overall_labels_map[label].union(label_pixel_map[label])
				
				else:    # Not classified
					res[row][col] = bw_img[row][col]
					if (label in overall_labels) == False:
						overall_labels_map[label] = label_pixel_map[label]
						
						for point in overall_labels_map[label]:
							overall_labels[point[0]][point[1]] = label

					else:
						max_key = 1
						for key in overall_labels_map:
							if key > max_key:
								max_key = key
						max_key += 1

						overall_labels_map[max_key] = overall_labels_map.pop(label)
						overall_labels_map[label] = label_pixel_map[label]
						
						for point in overall_labels_map[label]:
							overall_labels[point[0]][point[1]] = label
							#print("HOW MANY TIMES????")
							res[point[0]][point[1]] = bw_img[point[0]][point[1]]

						for point in overall_labels_map[max_key]:
							overall_labels[point[0]][point[1]] = max_key
		'''



		#print(overall_labels)

		if (127 in res) == False:
			if res[7,0] == 0:
				for i in range(29):
					for j in range(29):
						if i<=6 and (j<=6 or (j>=22 and j<=28)):   # upper left locator and upper right
							continue
						if (i>=22 and i<=28) and j<=6:  # lower left
							continue
						res[i][j] = abs(res[i][j] - 255)

			return res

	if res[7,0] == 0:
		for i in range(29):
			for j in range(29):
				if i<=6 and (j<=6 or (j>=22 and j<=28)):   # upper left locator and upper right
					continue
				if (i>=22 and i<=28) and j<=6:  # lower left
					continue
				res[i][j] = abs(res[i][j] - 255)
	return res



def preprocess(qr_list):
	res = np.ones([29, 29]) * 255
	res[0:7, 0] = 0
	res[0:7, 6] = 0
	res[0:7, -1] = 0
	res[0:7, -7] = 0

	res[0, 0:7] = 0
	res[0, -7:] = 0
	res[6, 0:7] = 0
	res[6, -7:] = 0

	res[-7:, 0] = 0
	res[-7:, 6] = 0
	res[-7, 0:7] = 0
	res[-1, 0:7] = 0

	res[2:5, 2:5] = 0
	res[2:5, -5:-2] = 0
	res[-5:-2, 2:5] = 0

	cnt = 0
	black_white_list = []
	for qr_key in qr_list:
		qr_img = qr_list[qr_key]
		black_white = np.ones([29, 29]) * 127

		index = np.zeros([29,29])
		#print("qr_img:", qr_img)

		for i in range(29):
			for j in range(29):
				if i<=6 and (j<=6 or (j>=22 and j<=28)):   # upper left locator and upper right
					continue
				if (i>=22 and i<=28) and j<=6:  # lower left
					continue
				block = qr_img[i*20:(i+1)*20, j*20:(j+1)*20]
				#print("block:",block)
				count_black = 0
				count_white = 0
				for m in range(20):
					for n in range(20):
						if block[m][n] == 0:
							count_black+=1
						elif block[m][n] == 255:
							count_white+=1

				if count_black / 400 >= 0.5:
					black_white[i,j] = 0
				elif count_white / 400 >= 0.3:
					black_white[i,j] = 255
				'''
				if qr_key == '13.png':
					for m in range(29):
						cv2.line(qr_img, (m*20, 0), (m*20, 580), 0, 2)
						cv2.line(qr_img, (0, m*20), (580, m*20), 0, 2)
					cv2.imshow('img', np.uint8(qr_img))
					cv2.waitKey(1)
				'''
		
		black_white[0:7, 0:7] = 127
		black_white[0:7, -7:] = 127
		black_white[-7:, 0:7] = 127

		black_white_list.append(black_white)

		cv2.imshow("bw", np.uint8(black_white))
		cv2.waitKey(1)
		print(qr_key)
		print(cv2.imwrite('./inspection/'+qr_key,black_white))
		cnt+=1

	return black_white_list


img_list = openimage("./image/")
qr_list = {}
index = 1
for img_key in img_list:
	img = img_list[img_key]
	qr_img = get_qrcode_and_type(img)
	if qr_img.size != 0:
		#qr_img=cv2.cvtColor(qr_img,cv2.COLOR_BGR2GRAY)
		#_,qr_img=cv2.threshold(qr_img,140,255,cv2.THRESH_BINARY)  #convert to binary
		qr_img = cv2.resize(qr_img, (580,580))
		qr_img = qr_img[5:575, 5:575]
		qr_img = cv2.resize(qr_img, (580,580))
		#print(qr_img)
		#for i in range(29):
		#	cv2.line(qr_img, (i*20,0), (i*20,579), 127, 2, 1)
		#	cv2.line(qr_img, (0,i*20), (579,i*20), 127, 2, 1)
		#qr_img = cv2.inRange(qr_img,(140,0,140),(255,180,255))


		mask1 = cv2.inRange(qr_img,(0,0,0),(255,180,255))
		mask2 = cv2.inRange(qr_img,(150,0,0),(255,255,255))
		mask3 = cv2.inRange(qr_img,(0,0,200),(255,255,255))

		img1 = cv2.bitwise_or(mask1, mask2, mask3) / 2

		img2 = cv2.inRange(qr_img, (170,0,170), (255,100,255))

		# img1: Black is green, white is others
		# img2: White is purple, black is others
		# How can I merge these imgs????????

		#img1 = cv2.bitwise_and(qr_img,qr_img, mask=mask)

		qr_img = img1 + img2/2
		#print(qr_img)

		#mask = cv2.inRange(qr_img, 0, 10)
		#qr_img = cv2.bitwise_and(qr_img,qr_img, mask=mask)

		

		
		#print(another_mask)
		#img2 = cv2.bitwise_and(qr_img, qr_img, mask=another_mask)

		#qr_img=cv2.cvtColor(qr_img,cv2.COLOR_BGR2GRAY)

		#for i in range(580):
		#	for j in range(580):
		#		#print(img1[i][j])
		#		if img1[i][j] == 0:
		#			qr_img[i][j] = 0
		#		elif img2[i][j] == 255:
		#			qr_img[i][j] = 255
		
		#print(qr_img)
		#cv2.imshow('img', qr_img)
		cv2.imshow('img', np.uint8(qr_img))
		cv2.waitKey(1)
		cv2.imwrite('./saved/'+img_key,qr_img)
		qr_list[img_key] = qr_img
		index+=1

black_white_list = preprocess(qr_list)
final_res = decode_bw_list(black_white_list)

cv2.imshow('img', final_res)
cv2.waitKey(1)
cv2.imwrite('./saved/final_res.png',final_res)

#cv2.imshow('img', res)
#cv2.waitKey(0)

#decoded = decode_qrcode(qr_list)
#cv2.imshow('img', decoded)
#cv2.waitKey(0)

#qr_img_list = openimage("saved")
