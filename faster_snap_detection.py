print("Importing libraries...")
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
# This is needed, for us to import utils since the utils is inside the object_detection folder.
sys.path.append("../models/research")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
print("Initializing paths...")
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, "IG", "frozen_inference_graph.pb")
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, "dataset", "label.pbtxt")
# path to where the capture images will be save
SAVE_IMAGES_PATH = os.path.join(CWD_PATH, 'save_images')
#Number of classes
NUM_CLASSES = 1


print("Loading the Inference Graph into memory, please wait...")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

detection_masks = detection_graph.get_tensor_by_name('detection_masks:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Initialize webcam feed
video = cv2.VideoCapture(0)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (masks, boxes, scores, classes, num) = sess.run(
        [detection_masks,detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
    # masks = utils_ops.reframe_box_masks_to_image_masks(
    #                    np.squeeze(masks), np.squeeze(boxes), 480, 640)
    # Draw the results of the detection (aka 'visulaize the results')
    
    masks = masks[0]
    masks = np.squeeze(masks).astype(np.uint8)
    result = np.reshape(masks, (-1, masks.shape[-1])).astype(np.float32).sum()
    print(result)
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        instance_masks=masks,
        use_normalized_coordinates=True,
        line_thickness=2)
    

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        print(masks)
        break

# Clean up
video.release()
cv2.destroyAllWindows()



