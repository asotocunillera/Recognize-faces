import face_recognition
import argparse
import pickle
import cv2 as cv
import os

info = { 'name': 'Encode Faces',
         'desc': 'Pre-train the network by constructing the 128d face embeddings for all faces in the dataset',
         'author': 'Álvaro Soto Cunillera',
         'email': 'asotocunillera@gmail.com',
         'year' : 2020,
         'version': [ 1, 0, 0, ],
         'license': 'Álvaro Soto Cunillera',
         }

# Argument parser
def get_args(info):

	parser = argparse.ArgumentParser(description='{desc}'.format(** info),epilog='{license}, {email}'.format(** info))

	# define argument validation
	parser.add_argument('-d','--dataset', required = True,
		help = 'path to dataset')
	parser.add_argument('-m','--method', type= str, default='cnn',
		help = 'face detection method tu use: "cnn" (default, recommended with gpu) or "hog" ')

	return vars(parser.parse_args())

def rename_files(person_path):

	# Rename images for easier processing
	person = os.path.basename(os.path.normpath(person_path))
	try:
		print(f'Preprocessing images of {person.replace("_"," ").title()}')
		for (i,name) in enumerate(os.listdir(person_path)):
			os.rename(os.path.join(person_path, name), os.path.join(person_path, ''.join([str(i), '.jpg'])))
	except:
		print(f'Files already preprocessed for {person.replace("_"," ").title()}')

def main():

	global info
	args = get_args(info)
	print('Starting process...')
	dataset_path = os.path.join('dataset', args['dataset'])

	people = os.listdir(dataset_path)

	known_encodings =  []
	known_names = []

	for person in people:
		person_path = os.path.join(dataset_path, person)
		rename_files(person_path)
		person_images = os.listdir(person_path)
		for (i,person_image) in enumerate(person_images):
			print(f'Processing image: {i+1}/{len(person_images)} of {person.replace("_"," ").title()}')

			image_path = os.path.join(person_path, person_image)
			img = cv.imread(image_path)
			rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) #dlib needs RGB images

			faces = face_recognition.face_locations(rgb, model = args['method'])

			encodings = face_recognition.face_encodings(rgb, faces)

			for encoding in encodings:
				known_encodings.append(encoding)
				known_names.append(person)

	#Save encodings in a pickle file for future recognition
	print('Serializing encodings...')
	encoding_path = os.path.join('encodings', f'{args["dataset"]}.pickle')
	data = {'encodings': known_encodings, 'names': known_names}
	f = open(encoding_path, "wb")
	f.write(pickle.dumps(data))
	f.close()

if __name__=='__main__':
	main()