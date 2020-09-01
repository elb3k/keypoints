import cv2
import os
import torch
import json
import numpy as np
import h5py
from tqdm import tqdm

KEYPOINTS = 17


def process(annotations_path, image_path, h5_path):
    """
    Process path
    """

    # Load annotations
    with open(annotations_path, "r") as f:
        raw = json.load(f)

    data = dict()


    # Process for each annotation
    for row in tqdm(raw["annotations"], desc="Processing annotations"):
        
        image_id = row["image_id"]
        keypoints = row["keypoints"]
        bbox = row["bbox"]

        if image_id not in data:

            data[image_id] = {"image": None, "data": [{ "keypoints": keypoints, "bbox":bbox }] }
        
        else:
            data[image_id]["data"].append({"keypoints": keypoints, "bbox": bbox})   


    # Get Image info to data
    for image_info in tqdm(raw["images"], desc="Processing images"):

        if image_info["id"] in data:
            data[image_info["id"]]["image"] = image_info


    with h5py.File(h5_path, "w") as hf:

        labels = np.empty( (0, KEYPOINTS*3), dtype=np.float)

        index = 0

        # Process every image
        for image_id, info in tqdm(data.items(), desc="Writing %s"%h5_path):

            image = cv2.imread(os.path.join(image_path, info["image"]["file_name"]))

            if image is None:
                continue

            for row in info["data"]:
                
                # BBox
                x, y, width, height = int(np.floor(row["bbox"][0])), int(np.floor(row["bbox"][1])), \
                                    int(np.ceil(row["bbox"][2])), int(np.ceil(row["bbox"][3])) 

                if width < 32 or height < 32:
                    continue

                # Cropped image
                crop = image[y:y+height, x: x+width]

                # # Resize image
                # crop = cv2.resize(image, (img_width, img_height))


                # Locations of keypoints
                keypoints = row["keypoints"]
                
                for i in range(0, KEYPOINTS*3, 3):

                    if keypoints[i+2] == 2:
                        keypoints[i] = (keypoints[i] - x) / width 
                        keypoints[i+1] = (keypoints[i+1] - y) / height

                        if (keypoints[i]>1.0 or keypoints[i+1]>1.0):
                            print( "keypoints[i]=%f, keypoints[i+1]=%f, x=%f, y=%f, width=%f, height=%f" % (keypoints[i], keypoints[i+1], x, y, width, height) )
                    else:
                        keypoints[i] = keypoints[i+1] = 0.0
                
                # Append label
                labels = np.append(labels, [keypoints], axis=0)

                # Store image in h5py
                hf.create_dataset(name="i"+str(index), data=crop, shape=(height, width, 3), compression="gzip", compression_opts=9)
                
                # Update index
                index = index + 1

        labels_dataset = hf.create_dataset(name="labels", data=labels, shape=labels.shape, dtype=np.float)

                

    return data


if __name__ == "__main__":

    process("dataset/annotations/person_keypoints_train2017.json", "dataset/train2017", "dataset/keypoints/train.h5")
