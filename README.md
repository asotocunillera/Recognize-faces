# Recognizing Faces

Python project to detect and recognize faces, from an image database or through captures taken from a live webcam video.

## How to use

This project can be used in different ways:
1. You can create a database of images extracted from captures of a webcam video, by using the script `face_dataset.py`.
2. You can include your own image database and train the tool to recognize faces in images, with `encode_faces.py`. This encoded faces will be saved in a pickle file.
3. With the encoded database of the last script, you can test different images to see if this tool could recognize faces with `image_recognition.py`.

## Requirements
This file is developed from opencv, dlib and face_recognition libraries so please, be sure to create a virtual environment and install all dependencies:
##### Create a virtual environment
`python -m venv venv`
##### Install requirements
`pip install -r requirements`

### Face Dataset
The way of run this script is:
`python face_dataset.py -o [name_people]`

If you want yo can type `python face_dataset.py -h` to read a brief description of the script.

When the file is running and the Webcam is open, it automatically detects all faces and if you type 'c' Key yo can save a screenshot in a dataset folder created automatically with the [name_people] you use as input. (*datset/[name_people]/[name_people]* )

### Encode Faces
The way of run this script is:
`python face_dataset.py -d [name_people] -m [hog_or_cnn]`

In this case, it has an optional argument * -m, --method * that identifies the face detection method. By default, this argument takes the cnn method, (which is recommended with gpu), due to its precision. However, without a gpu, the cnn method could take a long time to process, so I recommend hog method in this case.

All people into *datset/[name_people]* directory is read and are used to pre-train the network by constructing the 128d face embeddings for all faces in the dataset.

Encoded faces will be saved in a pickle file into the encoding_path folder, created automatically. (*encodings/[name_people.pickle]* )

### Image Recognition
The way of run this script is:
`python face_dataset.py -e [encoding_path] -i [image_to_test_path] -m [hog_or_cnn] `

This script has two argument and the optional argument to choose the face detection method.
- *-e, --encodings* : path to dataset of encoded faces.
- *-i, --image* : path to image to recognize

First the pickle data is read from [encoding_path] and then the test image is compared with the serialized ones to find if the people can be recognized or categorized as *Unknown*

## Autor

Álvaro Soto, asotocunillera@gmail.com
