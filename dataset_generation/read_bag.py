from pathlib import Path
from rosbags.highlevel import AnyReader

from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

from rosbags.image import message_to_cvimage
import cv2
import os


def ros2bag_to_images(bag_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    # create reader instance and open for reading
    with AnyReader([Path(bag_path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/color/image_raw']
        img_idx = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            img = message_to_cvimage(msg)

            cv2.imwrite(os.path.join(save_path, f"{img_idx:05d}_rgb.png"), img)

            img_idx += 1

def ros1bag_to_images(bag_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    # create reader instance
    with Reader(bag_path) as reader:
        # topic and msgtype information is available on .connections list
        # for connection in reader.connections:
        #     print(connection.topic, connection.msgtype)
            
        # iterate over messages
        img_idx = 0
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/wrist_camera/color/image_raw':
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)

                img = message_to_cvimage(msg)

                cv2.imwrite(os.path.join(save_path, f"{img_idx:05d}_rgb.png"), img)

                img_idx += 1

def main():

    bag_path = '/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_raw/real_bag'
    bag_dirs = [os.path.join(bag_path, file) for file in sorted(os.listdir(bag_path))]
    
    for bag_dir in bag_dirs:
        
        print(bag_dir)

        save_path = bag_dir.replace(".bag", "").replace("raw", "processed")

        try:
            ros1bag_to_images(bag_dir, save_path)
        except:
            print("Error opening bag file, corrupted")
            os.rmdir(save_path)


if __name__ == "__main__":
    main()