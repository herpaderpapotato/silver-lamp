import os
import json
import random
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm


class datasetloader(tf.keras.utils.Sequence):
    def __init__(self, labels_folder, images_folder, duration = 60, batch_size=32, croppct = 0.2, augment = False, image_size = 512):
        self.image_size = image_size
        self.labels_folder = labels_folder
        self.batch_size = batch_size
        self.duration = duration
        self.images_folder = images_folder
        self.allimagefiles = self.getallimagefiles()

        self.required_images = []
        self.dataset = self.init_set()
        self.allimages = self.get_allimages()

 
        #self.dataset_dict = self.build_set()
        self.labels = self.dataset.keys()
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()
        self.croppct = croppct
        self.augment = augment
        #self.image_dict = self.get_images()

    def getallimagefiles(self):
        allfiles =  glob(os.path.join(self.images_folder, '**', '*.jpg'), recursive=True)
        allfiles.extend(glob(os.path.join(self.images_folder + '_crop', '**', '*.jpg'), recursive=True))
        allfiles.extend(glob(os.path.join(self.images_folder + '_poi', '**', '*.jpg'), recursive=True))
        allfiles = sorted(allfiles)
        return allfiles

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))
    
    def __getitem__(self, index):
        batch_data = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        #print(batch_labels)
        batch_labels = []
        batch_features = []
        for batch_label in batch_data:
            #labelfile = random.choice(self.dataset_dict[batch_label])
            #label = json.load(open(labelfile))

            items = self.dataset[batch_label]
            cancrop = True
            canpoi = True
            for item in items:
                if item[2] == False:
                    cancrop = False
                if item[3] == False:
                    canpoi = False
                
            if cancrop:
                cancrop = random.choice([True, False])
            elif canpoi:
                canpoi = random.choice([True, False])

            keyindex = 0
            item_labels = []
            hflip = random.randint(0, 3)
            blur = random.randint(0, 3)
            if self.augment:
                if self.croppct > 0:
                    crop = random.randint(0, 6)
                else:
                    crop = 0
                lhcrop = random.uniform(0, 1)
                brighten = random.randint(0, 3)
                blocker = random.randint(0, 6) # 1/6 chance of blocker
                blockerpct = random.uniform(0.1, 0.5)
                blockerx = random.randint(0, int(self.image_size - self.image_size * blockerpct))
                blockery = random.randint(0, int(self.image_size - self.image_size * blockerpct))
                blockercolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                blackandwhite = random.randint(0, 6)
            else:
                crop = 0
                lhcrop = 0
                brighten = 0
                blocker = 0
                blockerpct = 0
                blockerx = 0
                blockery = 0
                blockercolor = (0, 0, 0)
                blackandwhite = 0
            if canpoi:
                crop = 0

            reverse = random.randint(0, 1)

            feature_frames = []
            feature_labels = []



            for item in items:
                #print(item)
                if cancrop:
                    #frame = self.images_crop[item[1]].copy()
                    frame = self.allimages[item[6]].copy()
                elif canpoi:
                    #frame = self.images_poi[item[1]].copy()
                    frame = self.allimages[item[5]].copy()
                else:
                    #frame = self.images[item[1]].copy()
                    frame = self.allimages[item[4]].copy()




                #cv2.imshow('frame', frame)
                #cv2.waitKey(160)
                if hflip == 1:
                    frame = cv2.flip(frame, 1)
                if blur == 1:
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)

                if self.augment:
                    if crop == 1:
                        originalx = frame.shape[1]
                        originaly = frame.shape[0]
                        vcrop = int(self.croppct * frame.shape[0])
                        hlcrop = int(lhcrop * self.croppct * frame.shape[1])
                        hrcrop = frame.shape[1] - hlcrop

                        frame = frame[vcrop:, hlcrop:hrcrop]
                        frame = cv2.resize(frame, (originalx, originaly))
                    if brighten == 1:
                        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
                    if blocker == 1:
                        topleft = (blockerx, blockery)
                        bottomright = (min(int(blockerx + self.image_size * blockerpct), self.image_size), min(int(blockery + self.image_size * blockerpct), self.image_size))
                        cv2.rectangle(frame, topleft, bottomright, blockercolor, -1)
                    if blackandwhite == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)



                if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                feature_frames.append(frame)
                feature_labels.append(item[0])
            if reverse == 1:
                feature_frames = feature_frames[::-1]
                feature_labels = feature_labels[::-1]
            batch_labels.append(feature_labels)
            batch_features.append(feature_frames)

        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)

        return batch_features, batch_labels
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
        # shuffle labels
        self.labels = list(self.labels)
        random.shuffle(self.labels)

    
    def get_allimages(self):
        allimages = []
        for imagefile in tqdm(self.allimagefiles):
            image = cv2.imread(imagefile).astype(np.uint8)
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            allimages.append(image)
        return allimages

            
    
    
    def init_set(self):
        label_folders = glob(os.path.join(self.labels_folder, '*'))
        label_folders = [label_folder for label_folder in label_folders if os.path.isdir(label_folder)]
        label_sequences = {}
        required_images = []
        allimagefiles = self.allimagefiles
        allimagefiles = [allimagefiles[:-4] for allimagefiles in allimagefiles]
        allimagefiles = [allimagefiles[len(self.images_folder):] for allimagefiles in allimagefiles]
        
        images_crop_folder = self.images_folder + '_crop'
        images_poicrop_folder = self.images_folder + '_poi'
        for label_folder in tqdm(label_folders):
            labels = glob(os.path.join(label_folder, '*.txt'))
            labels = sorted(labels, key=lambda x: int(os.path.basename(x)[:-4]))
            label_values = []
            for label in labels:
                with open(label) as f:
                    label_values.append(float(f.read()))
            labels = [label[:-4] for label in labels]
            labels = [label[len(self.labels_folder):] for label in labels]
            for i in tqdm(range(len(labels) - self.duration)):
                label = int(os.path.basename(labels[i]))
                #print(label)
                linear = True
                can_crop = True
                can_poi = True
                # can_double = True
                # can_crop_double = True
                for j in range(self.duration):
                    if label + j != int(os.path.basename(labels[i + j])):
                        linear = False
                        break
                for j in range(self.duration):
                    if not os.path.exists(os.path.join(self.images_folder, os.path.basename(label_folder), str(label + j) + '.jpg')):
                        linear = False
                        break
                for j in range(self.duration):
                    if not os.path.exists(os.path.join(images_crop_folder, os.path.basename(label_folder), str(label + j) + '.jpg')):
                        can_crop = False
                        break
                for j in range(self.duration):
                    if not os.path.exists(os.path.join(images_poicrop_folder, os.path.basename(label_folder), str(label + j) + '.jpg')):
                        can_poi = False
                        break

                
                if linear:
                    label_sequence = []
                    
                    for j in range(self.duration):
                        image_index = 0
                        poi_image_index = 0
                        crop_image_index = 0
                        # find image path in allimagefiles
                        if not can_crop:
                            crop_image_index = -1
                        if not can_poi:
                            poi_image_index = -1
                        # haslooped = False
                        track_index = 0
                        image_index = allimagefiles.index(labels[i+j])
                        if can_crop:
                            crop_image_index = allimagefiles.index('_crop' + labels[i+j])
                        if can_poi:
                            poi_image_index = allimagefiles.index('_poi' + labels[i+j])
                        

                            # if track_index == len(allimagefiles):
                            #     track_index = 0
                            #     haslooped = True 
                        #label_sequence.append([label_values[i + j], labels[i+j], can_crop, can_double, can_crop_double])
                        #print([label_values[i + j], labels[i+j],can_crop, can_poi, image_index, poi_image_index, crop_image_index, allimagefiles[image_index], allimagefiles[poi_image_index], allimagefiles[crop_image_index]])
                        label_sequence.append([label_values[i + j], labels[i+j],can_crop, can_poi, image_index, poi_image_index, crop_image_index])
                        required_images.append(labels[i+j])
                    label_sequences[labels[i]] = label_sequence
        self.required_images = required_images
        return label_sequences


if __name__ == '__main__':
    cwd = os.getcwd()
    labels_folder = cwd + '/dataset/universal_labels'
    images_folder = cwd + '/dataset/features512'
    gen = datasetloader(labels_folder,images_folder, duration = 60, batch_size=32, image_size=384)
    