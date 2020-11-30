import argparse
import time
import cv2 as cv
import os

info = { 'name': 'face dataset',
         'desc': 'Detecting faces and Creating a face dataset from WebCam Live Video',
         'author': 'Álvaro Soto Cunillera',
         'email': 'asotocunillera@gmail.com',
         'year' : 2020,
         'version': [ 1, 0, 0, ],
         'license': 'Álvaro Soto Cunillera',
         }

# Argument parsing
def get_args(info):

	parser = argparse.ArgumentParser(description='{desc}'.format(** info),epilog='{license}, {email}'.format(** info))

	# define argument validation
	parser.add_argument('-o','--output', required = True, 
		help=r'''path to the output directory: dataset\[name_chosen].
				If it does not exist it is created automatically''')

	# get command line arguments
	return vars(parser.parse_args())

def rescale_frame(frame, scale = 0.75):
	# Images, Videos, Live Video
	width = int(frame.shape[1]*scale)
	height = int(frame.shape[0]*scale)
	dimensions = (width, height)
	return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# Main
def main():

	global info
	# get command line arguments
	args = get_args(info)

	output_path = os.path.join('dataset', args['output'], args['output'])

	# Haar cascade
	haar_cascade = r'haar_cascade\haarcascade_frontalface_default.xml'
	face_cascade = cv.CascadeClassifier(haar_cascade)

	print('Opening WebCam...')
	cap = cv.VideoCapture(0)
	time.sleep(2)
	total = 0

	while True:
		flag ,frame = cap.read()
		img = rescale_frame(frame, scale = 0.9)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			gray, scaleFactor = 1.1, 
			minNeighbors = 4, minSize= (30,30))

		for (x,y,w,h) in faces:
			cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)

		cv.imshow('Press key "c" to take picture, "q" to exit', img)
		key = cv.waitKey(1) & 0xFF

		#if 'C' is pressed, save the frame into output directory
		if key == ord('c'):
			# if output path does not exist it is created
			if not os.path.isdir(output_path): os.makedirs(output_path)
			
			f = os.path.join(output_path, f'{str(total).zfill(5)}.png')
			print(f'Frame {str(total).zfill(5)} saved')
			cv.imwrite(f, frame)
			total+=1

		#if 'q' is pressed, exit the loop
		if key == ord('q'):
			break

	print( 'Closing WebCam...')
	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()

