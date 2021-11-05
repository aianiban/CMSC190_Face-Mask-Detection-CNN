import os
import cv2
import numpy as np
from tqdm import tqdm


REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data.

class MaskDetection():
    IMG_SIZE = 50
    with_mask = "data/with_mask"
    without_mask = "data/without_mask"
    TESTING = "data/Testing"
    LABELS = {without_mask: 0, with_mask: 1}
    training_data = []

    with_mask_count = 0
    without_mask_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.with_mask:
                            self.with_mask_count += 1
                        elif label == self.without_mask:
                            self.without_mask_count += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('With Mask:',maskdetect.with_mask_count)
        print('Without Mask:',maskdetect.without_mask_count)

if REBUILD_DATA:
    maskdetect = MaskDetection()
    maskdetect.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

print(training_data[20])

import matplotlib.pyplot as plt

plt.imshow(training_data[20][0])
plt.show()