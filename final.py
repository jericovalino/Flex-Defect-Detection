from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
import numpy
import numpy as np
import sys
import tensorflow as tf
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


class Application:

    def __init__(self,detection_graph):
        """ Initialize application which uses OpenCV + Tkinter. It displays
        a video stream in a Tkinter window and stores current snapshot on disk """
        self.cam = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.current_image = None  # current image from the camera
        self.root = Tk()  # initialize root window
        self.root.overrideredirect(True)

        # Gets both half the screen width/height and window width/height
        screenLengthX = int(self.root.winfo_screenwidth()/2 - (843/2))
        screenLengthY = int(self.root.winfo_screenheight()/2 - (744/2))

        self.root.geometry(f"843x744+{screenLengthX}+{screenLengthY}") # window position(center)

        self.root.title("")  # set window title
        self.root.resizable(False,False)
        self.root.configure(bg="#222222") #bg color
        self.root.protocol('DELETE_WINDOW', self.destructor)


        """declairing variable for UI starts here"""
        self.topBar = Frame(self.root)
        self.blueLine = Frame(self.root, bg="#4285F4", height=7)
        self.titleBar = Frame(self.root, bg="#222222", height=30)
        self.centerFrame = Frame(self.root, bg="#222222")
        self.terminalFrame = Frame(self.root, bg="#222222", height=260)
        self.panel = Label(self.centerFrame)  # initialize image panel
        self.lblFrame = LabelFrame(self.centerFrame,text="MENU",background="#f5f5f5")
        logoImg = PhotoImage(file = 'logo.PNG')
        logo = Label(self.titleBar, image=logoImg, bd=0, bg="#222222"); logo.image = logoImg
        """declairing variable for UI ends here"""


        """declaring variables for top bar buttons starts here"""
        exitImg = PhotoImage(file = "red.PNG")
        minImg = PhotoImage(file = "yellow.PNG")
        maxImg = PhotoImage(file = "green.PNG")
        exitbtn = Button(self.topBar, image=exitImg, bd=0, command=self.destructor); exitbtn.image = exitImg
        maxbtn = Button(self.topBar, image=maxImg, bd=0); maxbtn.image = maxImg
        minbtn = Button(self.topBar, image=minImg, bd=0, command=self.destructor); minbtn.image = minImg
        """declaring variables for top bar buttons ends here"""


        """initializing dragging feature starts here"""
        self.drag = self.topBar
        self.drag.bind('<ButtonPress-1>', self.StartMove)
        self.drag.bind('<ButtonRelease-1>', self.StopMove)
        self.drag.bind('<B1-Motion>', self.OnMotion)
        """initializing dragging feature ends here"""


        exitbtn.grid(column = 0, row = 0, padx=(3,0))
        minbtn.grid(column = 1, row = 0, padx=(1,0))
        maxbtn.grid(column = 2, row = 0, padx=(1,0))

        titleLabel = Label(self.titleBar, text="Cosmetic Quality Defect Detection in Electronics", bd=0, bg="#222222", foreground='#fefefe',font=('verdana', 10))

        self.topBar.grid(sticky=(W,E))        
        self.blueLine.grid(row=1,column=0,sticky=(W,E))
        self.titleBar.grid(row=2,column=0,sticky=(W,E))
        logo.grid(row=0, column=0, pady=3); titleLabel.grid(row=0, column=1, padx=10)
        self.centerFrame.grid(row=3, column=0,sticky=(W,E))
        self.terminalFrame.grid(row=4,column=0,sticky=(N,E,S,W),padx=3,pady=3)
        self.panel.grid(row=0,column=0,sticky=W,padx=(3,0))
        self.lblFrame.grid(row=0,column=1,sticky=(N,E,S,W),padx=(0,3))


        # create a button, that when pressed, will take the current frame and save it to file
        detectbtn = Button(self.lblFrame, text="Run Detection", command=self.take_snapshot)
        detectbtn.configure(foreground="white", background="#4285F4", width=20, height=2, relief=GROOVE, font=('verdana', 10, 'bold') )
        detectbtn.grid(column = 0, row = 1, sticky = (E,W))


        # create a button, that when pressed, will take the current frame and save it to file
        self.entr = Entry(self.lblFrame, text="evaluate")
        self.entr.configure(bd=1, background="#9e9e9e")
        self.entr.grid(column = 0, row = 0, sticky = (E,W))


        self.counter = 0

        # start a self.video_loop that constantly pools the video sensor for the most recently read frame
        self.video_loop()


    """functions for dragging feature starts here"""
    def StartMove(self, event):
        self.x = event.x
        self.y = event.y

    def StopMove(self, event):
        self.x = None
        self.y = None

    def OnMotion(self, event):
        x = (event.x_root - self.x - self.drag.winfo_rootx() + self.drag.winfo_rootx())
        y = (event.y_root - self.y - self.drag.winfo_rooty() + self.drag.winfo_rooty())
        self.root.geometry(f"+{x}+{y}")
    """functions for dragging feature ends here"""


    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.cam.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            frametk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.frametk = frametk  # anchor frametk so it does not be deleted by garbage-collector
            self.panel.config(image=frametk)  # show the image
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds


    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.cam.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application
    

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    def run_inference(self,rawimage,image):
        with detection_graph.as_default():
            with tf.Session() as sess:
                #os.system('mode con: cols=80 lines=4')
                os.system('cls')
                print("Click 'Run Detection' to capture.")
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
                cv2.imshow("result", frame)


    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        self.counter += 1
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}_Image_{}.jpg".format(ts.strftime("%Y_%m_%d"),self.counter)  # construct filename
        path = os.path.join(CWD_PATH, 'save_images', filename)  # construct output path
        image = cv2.cvtColor(numpy.array(self.current_image), cv2.COLOR_RGB2BGR)  # save image as jpeg file
        cv2.imwrite(path, image)
        result = self.entr.get()
        print(result)
        print("[INFO] saved! {}".format(filename))
                    
        CAPTURED_IMAGE_PATH = os.path.join(SAVE_IMAGES_PATH, filename)
        imageopen = Image.open(CAPTURED_IMAGE_PATH)
        print(CAPTURED_IMAGE_PATH)
        rawimage = cv2.imread(CAPTURED_IMAGE_PATH)
        image_np = self.load_image_into_numpy_array(imageopen)
        self.run_inference(rawimage,image_np)
    
    



if __name__ == "__main__":
    # start the app
    print("[INFO] starting...")
    run = Application(detection_graph)
    run.root.mainloop()
