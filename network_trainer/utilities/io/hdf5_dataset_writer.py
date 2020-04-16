import os
import h5py
import numpy as np

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # Check if data already exists
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", outputPath)

        # Open HDF5 database and create 2 datasets: features and labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # store buffer size and initialize it
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add rows and labels to buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check if buffer needs to be flushed
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write buffers to disk and reset them
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # store class labels in separate file
        dt = h5py.special_dtype(vlen=np.unicode)
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # flush if there are any leftovers
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()
