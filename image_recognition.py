import face_recognition
import argparse
import pickle
import cv2 as cv
import os

info = {
    'name': 'Image Recognition',
    'desc': 'Face recognition from comparison with encoded images',
    'author': 'Álvaro Soto Cunillera',
    'email': 'asotocunillera@gmail.com',
    'year': 2020,
    'version': [1,0,0],
    'license': 'Álvaro Soto Cunillera',
}
# Argument parser
def get_args(info):

    parser = argparse.ArgumentParser(description='{desc}'.format(** info))

    # define argument validation
    parser.add_argument('-e','--encodings', required = True,
        help = 'path to encoded faces dataset')
    parser.add_argument('-i','--image', required = True,
        help = 'path to image to recognize')
    parser.add_argument('-m','--method', type= str, default='cnn',
        help = 'face detection method tu use: "cnn" (default) or "hog" ') #cnn slower without a GPU

    return vars(parser.parse_args())

def main():

    global info
    args = get_args(info)

    print('Loading faces...')
    encoded_path = os.path.join('encodings', args['encodings'])
    pickle_data = pickle.loads(open(f'{encoded_path}.pickle', 'rb').read())
    img_path = os.path.join('test', args['encodings'], args['image'])

    img = cv.imread(img_path)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) #dlib needs RGB images

    print('Starting recognition...')
    faces = face_recognition.face_locations(rgb, model = args['method'])
    encodings = face_recognition.face_encodings(rgb, faces)

    names = []

    for encoding in encodings:

        name = 'Unknown' #Initialize the face as unknown

        #Compare test face with dataset
        matches = face_recognition.compare_faces(pickle_data['encodings'], encoding)

        if True in matches:
           #Grab all matches in order to chose the person with more matches 

           match_count = {}
           match_indexes = [i for (i,match) in enumerate(matches) if match]

           for i in match_indexes:
               name = pickle_data['names'][i]
               #Update nº of matches for that person
               match_count[name] = match_count.get(name,0)+1

            name = max(match_count, key=match_count.get)
        names.append(name)

        for (top, right, bottom, left), name in zip(faces, names):
            cv.rectangle(img, (left,top), (right, bottom), (255,0,0), 2)
            cv.putText(img, name.replace("_", " ").title(), (left, top-10), cv.FONT_HERSHEY_TRIPLEX, .5, (255,0,0),2)

        cv.imshow(name.replace("_", " ").title(), img)
        cv.waitKey(0)

    if __name__ == '__main__':
        main()