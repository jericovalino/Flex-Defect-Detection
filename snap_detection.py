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
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_app(detection_graph):
    with detection_graph.as_default():
        with tf.Session() as sess:
            os.system('mode con: cols=80 lines=4')
            os.system('cls')
            print("Press 'space' to capture.")

            def run_inference(rawimage,image,counter):
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                        real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                        real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                # Visualization of the results of a detection.
                frame = vis_util.visualize_boxes_and_labels_on_image_array(
                    rawimage,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=2)
                print(">>>showing evaluated image>>>")
                print(">>>result appears here!!!!>>>")
                
                while True:
                    cv2.imshow("test", frame)
                    k = cv2.waitKey(1)
                    if k % 256 == 27:
                        # ESC presssed
                        img_name = f"captured_{counter}_evaluated.png"
                        cv2.imwrite(os.path.join(SAVE_IMAGES_PATH, img_name), frame)
                        print(f"{img_name} saved!")
                        break
                counter += 1
                return counter

            # initializing the camera
            cam = cv2.VideoCapture(0)
            cv2.namedWindow("test")
            counter = 0
            while True:
                ret, frame = cam.read()
                cv2.imshow("test", frame)
                if not ret:
                    break
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k % 256 == 32:
                    # SPACE pressed
                    # Saving Captured Image
                    img_name = f"captured_{counter}.png"
                    cv2.imwrite(os.path.join(SAVE_IMAGES_PATH, img_name), frame)
                    print(f"{img_name} saved!")
                    CAPTURED_IMAGE_PATH = os.path.join(SAVE_IMAGES_PATH, f'captured_{counter}.png')
                    imageopen = Image.open(CAPTURED_IMAGE_PATH)
                    rawimage = cv2.imread(CAPTURED_IMAGE_PATH)
                    image_np = load_image_into_numpy_array(imageopen)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    counter = run_inference(rawimage,image_np,counter)
            cam.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app(detection_graph)

