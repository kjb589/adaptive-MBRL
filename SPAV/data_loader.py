import os
import numpy as np
import tensorflow as tf
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from tqdm import tqdm


# Is it ready
# yes


class DataLoader:
    def __init__(self, dataroot, version='v1.0-mini', sequence_length=8, image_size=(128, 128)):
        self.dataroot = dataroot
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.ood_scene_tokens = set([
            self.nusc.scene[4]['token'],  # OOD example scene
            self.nusc.scene[8]['token'],  # Another OOD example
        ])

    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size)
        return np.array(img) / 255.0  # Normalize to [0, 1]

    def get_sequences_and_labels(self):
        sequences = []
        labels = []

        for scene in tqdm(self.nusc.scene, desc="Loading Scenes"):
            scene_token = scene['token']
            label = 1 if scene_token in self.ood_scene_tokens else 0

            sample_token = scene['first_sample_token']
            images = []

            while len(images) < self.sequence_length:
                sample = self.nusc.get('sample', sample_token)
                cam_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])

                img_path = os.path.join(self.dataroot, cam_data['filename'])
                image = self.load_image(img_path)
                images.append(image)

                if cam_data['next'] == '':
                    break
                sample_token = self.nusc.get('sample_data', cam_data['next'])['sample_token']

            if len(images) == self.sequence_length:
                sequences.append(np.stack(images))  # Shape: (SEQ_LEN, H, W, C)
                labels.append(label)

        return np.array(sequences), np.array(labels)


if __name__ == "__main__":
    DATA_ROOT = "C:/Users/kjbut/MAVS-Examples/SPAV/nuScenes/"
    loader = NuScenesSequenceLoader(DATA_ROOT, sequence_length=8, image_size=(128, 128))

    X, y = loader.get_sequences_and_labels()
    print("Dataset shape:", X.shape)  # (N, SEQ_LEN, H, W, C)
    print("Labels shape:", y.shape)   # (N,)
    print("ID samples:", np.sum(y == 0))
    print("OOD samples:", np.sum(y == 1))

    # Save to disk if needed
    # np.save("X.npy", X)
    # np.save("y.npy", y)
