#!python3
try:
    import platform
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    import pickle
    import h5py
    import os
except ImportError as e:
    print(e)
    print("One or more dependencies missing!\nOpen README file to check required dependencies.")
    if platform.system() == 'Linux':
        print("\nYou can install all dependencies using install_dependencies.sh")
    else:
        print("\nYou can install all dependencies using install_dependencies.bat")
    quit()

# set&check paths for database & output
databasePath = r".{}database".format(os.path.sep)
if not os.path.exists(databasePath):
    os.mkdir(databasePath)
    print("Features file at {}\nDoes not exist!\nQuitting now".format(databasePath))
    quit()

# search for features file
file_found = False
feature_files_number = 0
database_list_files = os.listdir(databasePath)
for file_path in database_list_files:
    if 'features' in file_path and 'hdf5' in file_path:
        feature_files_number += 1
        databasePath += os.path.sep
        databasePath += file_path
        file_found = True
if not file_found:
    print("Features file at {}\nDoes not exist!\nQuitting now".format(databasePath))
    quit()
if feature_files_number != 1:
    print("There can be only 1 features file in database directory while there are {}\nRemove excessive files".format(feature_files_number))
    quit()

modelPath = r'.{}KrakN_model.cpickle'.format(os.path.sep)
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

