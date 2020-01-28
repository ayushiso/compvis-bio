import os
import cv2

for subdir, dirs, files in os.walk("Hela"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):
            image= cv2.imread(filepath)
            cv2.imwrite(filepath, image)
            print(filepath)