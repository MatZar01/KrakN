#!python3
# mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import h5py
import os

# set&check paths for database & output
databasePath = r"/content/gdrive/My Drive/KrakN/database/features_s_2.hdf5"
modelPath = r"/content/gdrive/My Drive/KrakN/KrakN.cpickle"

if not os.path.exists(databasePath):
    print("Features file at {}\nDoes not exist!\nQuitting now".format(databasePath))
    quit()
if os.path.exists(modelPath):
    os.remove(modelPath)
jobs = 1

# open database
db = h5py.File(databasePath, "r")
# and set the training / testing split index
i = int(db["labels"].shape[0] * 0.75)

# train Logistic Regression classifier
print("Tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(max_iter=700), params, cv=3, n_jobs=jobs, verbose=20)
model.fit(db["features"][:i], db["labels"][:i])
print("Best hyperparameter: {}".format(model.best_params_))

# evaluate model
print("Evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

# save model to disk
print("Saving model...")
f = open(modelPath, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()

