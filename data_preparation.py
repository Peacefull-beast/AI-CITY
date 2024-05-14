import pandas as pd
import numpy as np
import os
import cv2
import random
import shutil


# read data
df = pd.read_csv('gt_data.csv')

#append bounding box to each row
def retrieve_bounding_box(i):
    width = df.width[i]
    height = df.height[i]
    xc = df.left[i] + int(width / 2)
    yc = df.top[i] + int(height / 2)
    cls = df.cls[i]

    # Normalize coordinates
    width_norm = width / 1920
    xc_norm = xc / 1920
    yc_norm = yc / 1080
    height_norm = height / 1080

    bbox = [cls, xc_norm, yc_norm, width_norm, height_norm]
    return bbox  # Return the bbox values

df['bbox'] = [retrieve_bounding_box(i) for i in range(len(df))]


#generate labels
fmt=["%d", "%f", "%f", "%f", "%f"]


for i in range(len(df)):
    label = df.bbox[i]
    yolo_label_data = np.array(label)
    name = f"{df.vedio_id[i]}frame{df.frame[i]}.txt"
    name = os.path.join("labels", name)

    # Check if the file already exists
    if os.path.isfile(name):
        with open(name, 'a') as file:
            # Append the bounding box coordinates as a new line in the file
            np.savetxt(file, yolo_label_data.reshape(1, -1), fmt=fmt)

    else:
        # Create a new file and write the bounding box coordinates
        np.savetxt(name, yolo_label_data.reshape(1, -1), fmt=fmt)

print("Labels files generated successfully.")



#generating imagess
def extract_frames(video_path, output_folder, video_name):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 1
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame to a file
        frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}frame{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames extracted and saved to '{output_folder}'.")

#give path to your vedios folder
vedio_folder = 'videos'

vedio_list = os.listdir(vedio_folder)

i = 1
for video in vedio_list:
    video_path = os.path.join(vedio_folder,video)
    #path to your images folder where all images will be stored
    output_folder = "images"

    extract_frames(video_path, output_folder, video)
    print(f"{i} vedio done")
    i+=1


# Set the paths to your image and label folders
image_folder = 'images'
label_folder = 'labels'

# Get the list of image and label files
image_files = os.listdir(image_folder)
label_files = os.listdir(label_folder)

# Extract the filenames without extensions from label files
label_file_names = [os.path.splitext(label)[0] for label in label_files]

# Identify images without corresponding labels
images_without_labels = [image for image in image_files if os.path.splitext(image)[0] not in label_file_names]

# Delete images without labels
for image in images_without_labels:
    image_path = os.path.join(image_folder, image)
    os.remove(image_path)
    print(f"Deleted: {image_path}")

print("Images without corresponding labels have been deleted.")


# Set the paths to your image and label folders
image_folder = 'images'
label_folder = 'labels'

# Create train and validation folders
train_image_folder = 'the_ai_city\\images\\train'
train_label_folder = 'the_ai_city\\labels\\train'
validation_image_folder = 'the_ai_city\\images\\validation'
validation_label_folder = 'the_ai_city\\labels\\validation'

os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(validation_image_folder, exist_ok=True)
os.makedirs(validation_label_folder, exist_ok=True)

# Get the list of image files and shuffle them
image_files = os.listdir(image_folder)
random.shuffle(image_files)

# Split the files into train and validation sets (80% train, 20% validation)
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)
train_images = image_files[:split_index]
validation_images = image_files[split_index:]

# Move images and corresponding labels to train and validation folders
for image_file in train_images:
    image_path = os.path.join(image_folder, image_file)
    label_file = image_file.replace('.jpg', '.txt')  # Assuming labels have the same name but with .txt extension
    label_path = os.path.join(label_folder, label_file)
    shutil.move(image_path, train_image_folder)
    shutil.move(label_path, train_label_folder)

print("Training data moved to speicified location successfully")

for image_file in validation_images:
    image_path = os.path.join(image_folder, image_file)
    label_file = image_file.replace('.jpg', '.txt')  # Assuming labels have the same name but with .txt extension
    label_path = os.path.join(label_folder, label_file)
    shutil.move(image_path, validation_image_folder)
    shutil.move(label_path, validation_label_folder)

print("Validation data moved to speicified location successfully")

print("Data split into train and validation sets successfully.")

