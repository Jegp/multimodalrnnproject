import cv2
import os
import librosa
import librosa.display
from sklearn.externals import joblib
import numpy as np
#from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as PCA
import warnings
from time import time

warnings.filterwarnings("ignore", category=DeprecationWarning) 
n_subjects = 30
n_files = 40
n_frames = 50

t00 = time()
def showimage(image):
    cv2.imshow('image',image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Load sound file
def loadsound(file):
    y, sr = librosa.load(file, duration=1.1) # y = lyd og sr = samplerate
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20) #  20 mel-frequency cepstral coefficients
    return mfcc

# Load video file
def loadvideo(file):
    capture = cv2.VideoCapture(file)
    img_set = [capture.read()[1] for i in range(n_frames)]
    return img_set

# Returns data in tuple format
def loaddata():
    sound_data = []
    video_data = []
    subjects = [folder for folder in os.listdir('videos') if folder[0] != '.'] # List of subject names
    for subject in subjects[:n_subjects]:
        sounds = [loadsound('sound/'+subject+'/'+wav) for wav in os.listdir('sound/'+subject)[:n_files]]
        videos = [loadvideo('videos/'+subject+'/'+mpg) for mpg in os.listdir('videos/'+subject)[:n_files]]
        sound_data.append(sounds)
        video_data.append(videos)
    return (sound_data,video_data)

t0 = time()
print('Loading data set')
sound_data, video_data = loaddata()
print("done in %0.3fs" % (time() - t0))

t0 = time()
print('\nSplitting into train/test')
# Train and test set creation
video_train = [[video for video in person[0::2]] for person in video_data]
video_test =  [[video for video in person[1::2]] for person in video_data]
sound_train = [[video for video in person[0::2]] for person in sound_data]
sound_test =  [[video for video in person[1::2]] for person in sound_data]
print("done in %0.3fs" % (time() - t0))


t0 = time()
print('\nTraining eigenface model')
# Load sounds, create train data for eigenface fit
all_frames = [[np.array(frame).flatten() for video in person for frame in video] for person in video_train]
all_frames = [video for person in all_frames for video in person]
all_frames = [all_frames[x:x+100] for x in range(0, len(all_frames), 100)]


# Transform image vectors of the train and test into eigenface vectors
def create_eigenfaces(chunks):
    n_components = 45
    h = 288 # height of images
    w = 360 # width of images
    pca = PCA(n_components=n_components,whiten=True)
    count = 0
    for chunk in chunks:
        count+=1
        t1 = time()
        pca.partial_fit(chunk)
        print('Chunk ',count,'done in %0.3fs' % (time() - t1))
    
    pca.components_.reshape((n_components, h, w, 3)) # 3 with Colors, otherwise leave blank after width
    return pca

# Create eigenface model, create feature vectors
eigenfaces = create_eigenfaces(all_frames)

# Transform test and train images with PCA model
eigen_train = [[np.array([eigenfaces.transform(np.array(frame).flatten()) for frame in video]).flatten() for video in person] for person in video_train]
eigen_test = [[np.array([eigenfaces.transform(np.array(frame).flatten()) for frame in video]).flatten() for video in person] for person in video_test]
print("done in %0.3fs" % (time() - t0))



t0 = time()
print('\nCreate X and y for train and test in sound and video')
# Create X and y for train and test in sound and video
X_train_pca = [video for person in eigen_train for video in person]
X_test_pca = [video for person in eigen_test for video in person]

X_train_sound = [np.array(video).flatten() for person in sound_train for video in person]
X_test_sound = [np.array(video).flatten() for person in sound_test for video in person]

y_train = [idx for idx,person in enumerate(eigen_train) for video in person]
y_test = [idx for idx,person in enumerate(eigen_test) for video in person]
print("done in %0.3fs" % (time() - t0))

# Gather vectors in tuples and dumps
video_final = (X_train_pca, X_test_pca, y_train, y_test)
sound_final = (X_train_sound, X_test_sound, y_train, y_test)

# For at loade den gemte data, kør:
## video_data, sound_data = joblib.load('vector_data.pkl')
joblib.dump((video_final,sound_final), 'vector_data.pkl')
print("\nTotal time %0.3fs" % (time() - t00))
print(np.array(X_train_pca).shape)
