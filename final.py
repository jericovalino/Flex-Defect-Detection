"""
Naming Style Guide:
  variable,my_variable : variable
  function,my_function : function
  widget,myWidget      : tkinter widgets
  CONSTANT,MY_CONSTANT : constant
"""

from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import threading
from PIL import Image, ImageTk
import tkinter as tk
from datetime import datetime
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
# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_frozen_graph = os.path.join(CWD_PATH, "IG", "frozen_inference_graph.pb") # may change in runtime.
PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, "IG", "frozen_inference_graph.pb") # default(CONSTANT).
# List of the strings that is used to add correct label for each box.
path_to_labels = os.path.join(CWD_PATH, "dataset", "label.pbtxt") # may change in runtime.
PATH_TO_LABELS = os.path.join(CWD_PATH, "dataset", "label.pbtxt") # default(CONSTANT).
# Path to where the capture images will be save.
SAVE_IMAGES_PATH = os.path.join(CWD_PATH, 'save_images')
# Number of classes.
NUM_CLASSES = 1
# Grabs the current timestamp.
ts = datetime.now()


class Application:

    def __init__(self):
        """Initialize application which uses OpenCV + Tkinter"""
        self.cam = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera.
        self.current_frame = None  # current image from the camera.
        self.root = Tk()  # initialize root window.
        self.root.overrideredirect(True) # removes os default window border.
        self.root.attributes("-topmost", True) # makes the window always ontop.

        self.SAVING_DIR = os.path.join(CWD_PATH, "save_images")
        bg_color = "#222"

        # Gets both half the screen width/height and window width/height.
        screen_length_x = int(self.root.winfo_screenwidth()/2 - (848/2))
        screen_length_y = int(self.root.winfo_screenheight()/2 - (665/2))

        self.root.geometry(f"848x665+{screen_length_x}+{screen_length_y}") # window position(center).

        self.root.iconbitmap(default = 'icon.ico') # set icon.
        self.root.title("Flex Defect Detection")  # set window title.
        self.root.configure(bg = bg_color) # bg color
        self.root.protocol('DELETE_WINDOW', self.close_window)


        """declairing variable for GUI starts here"""
        self.topBar = Frame(self.root)
        self.blueLine = Frame(self.root, bg = "#4285F4", height = 7)
        self.titleBar = Frame(self.root, bg = bg_color, height = 30)
        self.centerFrame = Frame(self.root, bg = bg_color)
        self.terminalFrame = Frame(self.root, bg = bg_color, height = 100)
        self.terminalScrollBar = Scrollbar(self.terminalFrame)
        self.terminalListBox = Listbox(self.terminalFrame, bg="#1c313a", fg="#fff", width=103, height=8, borderwidth=0,
            highlightthickness = 0, font = ('verdana', 9), yscrollcommand=self.terminalScrollBar.set)
        self.terminalScrollBar.config(command = self.terminalListBox.yview)
        self.panel = Label(self.centerFrame, width = 91, bg = "#eee", text = "No signal...")  # initialize video panel.
        self.menuFrame = LabelFrame(self.centerFrame, text = "MENU", bg = "#eee")
        logo_img = PhotoImage(file = 'logo.PNG')
        logo = Label(self.titleBar, image = logo_img, bd = 0, bg = bg_color); logo.image = logo_img
        """declairing variable for UI ends here"""


        """declaring variables for top bar buttons starts here"""
        exit_img = PhotoImage(file = "red.PNG")
        min_img = PhotoImage(file = "yellow.PNG")
        max_img = PhotoImage(file = "green.PNG")
        exitButton = Button(self.topBar, image = exit_img, bd=0, command = self.close_window); exitButton.image = exit_img
        maxButton = Button(self.topBar, image = max_img, bd=0, command = self.clear_terminal); maxButton.image = max_img
        minButton = Button(self.topBar, image = min_img, bd=0, command = self.minimize); minButton.image = min_img
        """declaring variables for top bar buttons ends here"""


        """initializing dragging feature starts here"""
        self.drag = self.topBar
        self.drag.bind('<ButtonPress-1>', self.start_move)
        self.drag.bind('<ButtonRelease-1>', self.stop_move)
        self.drag.bind('<B1-Motion>', self.on_motion)
        """initializing dragging feature ends here"""

        # Declaring a Label variable that holds the title text.
        titleLabel = Label(self.titleBar, text="Cosmetic Quality Defect Detection in Electronics",
            bd=0, bg=bg_color, fg='#fefefe',font=('verdana', 10))

        self.topBar.grid(sticky = (W,E))
        exitButton.grid(column = 0, row = 0, padx = (3,0))
        minButton.grid(column = 1, row = 0, padx = (1,0))
        maxButton.grid(column = 2, row = 0, padx = (1,0))     
        self.blueLine.grid(row = 1, column = 0, sticky = (W,E))
        self.titleBar.grid(row = 2, column = 0, sticky = (W,E))
        logo.grid(row = 0, column = 0, pady = 3)
        titleLabel.grid(row = 0, column = 1, padx = 10)
        self.centerFrame.grid(row = 3, column = 0, sticky = (W,E))
        self.panel.grid(row = 0, column = 0, sticky = (N,S,W), padx = (3,0))
        self.menuFrame.grid(row = 0, column = 1, sticky = (N,E,S,W), padx = (0,3))
        self.terminalFrame.grid(row = 4, column = 0, sticky = (W,E))
        self.terminalListBox.grid(row = 0, column = 0, sticky = (N,E,S,W), padx = (3,0), pady = 3)
        self.terminalScrollBar.grid(row = 0, column = 1, sticky = (N,S,E), padx = (1,3), pady = 3)


        # creates a button, that when pressed, will take the current frame and then run the detection.
        self.detectionButton = Button(self.menuFrame, text = "RUN DETECTION", bg = "#e0e0e0",
            width = 23, height = 2, relief = GROOVE, font = ('verdana', 10), command = self.take_snapshot)
        self.detectionButton.grid(column = 0, row = 1, sticky = (E,W))

        # creates a button, that when pressed, will open a fileDialog for user to select saving dir.
        changeDirButton = Button(self.menuFrame, text = "change saving directory",
            relief=GROOVE, command = (self.change_saving_dir))
        changeDirButton.grid(column = 0, row = 3, sticky = (E,W), pady=(295,0))

        # creates a button, that when pressed, will call the funtion self.change_model.
        selectModelButton = Button(self.menuFrame, text = "change trained model",
            relief=GROOVE, command = (self.change_model))
        selectModelButton.grid(column = 0, row = 4, sticky = (E,W))

        # creates a button, that when pressed, will call the funtion self.change_label.
        selectLabelButton = Button(self.menuFrame, text = "change label file",
            relief=GROOVE, command = (self.change_label))
        selectLabelButton.grid(column = 0, row = 5, sticky = (E,W))

        # creates a button, that when pressed, will call the funtion self.load_defaults.
        restoreDefaultButton = Button(self.menuFrame, text = "restore default settings",
            relief = GROOVE, command = (self.load_defaults))
        restoreDefaultButton.grid(column = 0, row = 6, sticky = (E,W))
        
        # create a progress bar for detection process.
        self.progressBar = ttk.Progressbar(self.menuFrame, mode = 'determinate',
            orient = HORIZONTAL, maximum = 4, value = 0)
        self.progressBar.grid(column = 0, row = 0, sticky = (E,W))


        self.counter = 0 #counter for captured image numbering.
        self.listCounter = 1 #counter for terminal listbox (index).

        self.video_loop() # start a self.video_loop .

        self.root.bind('<Map>', self.check_map) # added bindings to pass windows status to function.
        self.root.bind('<Unmap>', self.check_map)

        self.load_defaults() # load default setting at start-up.

    def check_map(self, event):
        """apply override on deiconify"""
        if str(event) == '<Map event>':
            self.root.overrideredirect(True)

    def start_move(self, event):
        """commputes the changes in x and y when top bar is being drag"""
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        """resets x and y value when the mouse stops moving"""
        self.x = None
        self.y = None

    def on_motion(self, event):
        """change the position of the window/s based on the calculation when top bar is being drag"""
        x = (event.x_root - self.x - self.drag.winfo_rootx() + self.drag.winfo_rootx())
        y = (event.y_root - self.y - self.drag.winfo_rooty() + self.drag.winfo_rooty())
        self.root.geometry(f"+{x}+{y}")
        try:
            self.win.attributes('-topmost',True)
            self.win.geometry(f"+{x-5}+{y+23}")
        except:
            pass

    def video_loop(self):
        """get frame from the video stream and show it in panel every 30 milliseconds"""
        ok, self.frame = self.cam.read()  # read frame from video stream.
        if ok:  # frame captured without any errors.
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA.
            self.current_frame = Image.fromarray(cv2image)  # convert image for PIL.
            frametk = ImageTk.PhotoImage(image=self.current_frame)  # convert image for tkinter.
            self.panel.frametk = frametk  # anchor frametk so it does not be deleted by garbage-collector.
            self.panel.config(image = frametk, width = 640, height = 480)  # show the image in panel.
        if not ok:
            frametk = None
            self.panel.frametk = None
            self.clear_terminal()
            self.terminal_print("WARNING", "No connected camera found!!!")
            self.cam = cv2.VideoCapture(0)
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds.

    def terminal_print(self, mtype, message):
        """print something in the terminal(terminalListBox)"""
        time = datetime.now()
        self.terminalListBox.insert(self.listCounter, f"[{time.hour}:{time.minute}:{time.second}] [{mtype}] {message}")
        self.terminalListBox.see(END)
        self.listCounter += 1

    def clear_terminal(self):
        """clears the terminal(terminalListBox)"""
        self.terminalListBox.delete(0, END)
        self.progressBar.config(value = 0) # resets the progress bar

    def close_window(self):
        """Destroy the window and release all resources """
        self.root.destroy() # destroys window.
        self.cam.release() # release web camera.

    def minimize(self):
        """minimize the window"""
        self.root.wm_withdraw()
        self.root.wm_overrideredirect(False)
        self.root.wm_iconify()

    def load_inference_graph(self,path_to_frozen_graph):
        """load the (.pb) file into memory"""
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.terminal_print("INFO", "Loading the Inference Graph into memory. Done")

    def load_label(self,path_to_labels):
        """load the (.pbtxt) file into memory"""
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.terminal_print("INFO", "Loading labels into memory. Done.")

    def take_snapshot(self):
        """Take snapshot, then saves the captured to saving dir"""
        self.counter += 1
        self.detectionButton.config(state = DISABLED)
        try:
            self.close_result()
        except:
            pass

        captured_image_name = "{}_[{}]_Captured.jpg".format(ts.strftime("%d-%m-%y"),self.counter)  # construct filename
        captured_image_path = os.path.join(self.SAVING_DIR, captured_image_name)  # construct output path
        cv2.imwrite(captured_image_path, self.frame)
        self.terminal_print("SAVED", captured_image_name)

        imageopen = Image.open(captured_image_path)
        raw_image = cv2.imread(captured_image_path)
        image_np = self.load_image_into_numpy_array(imageopen)
        t1 = threading.Thread(target=self.run_inference, args=(raw_image,image_np)).start() # run inference in seperate thread to avoid mainloop freeze
        self.progressBar.config(value=1)

    def load_image_into_numpy_array(self,image):
        """loads images into numpy array that will be feed to the model"""
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference(self,raw_image,image):
        """runs the detection"""
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                self.terminal_print("INFO", "Evaluating captured image. Please wait...")                
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

                # add a progress on the progress bar
                self.progressBar.config(value = 2)
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict = {image_tensor: np.expand_dims(image, 0)})
                # add a progress on the progress bar
                self.progressBar.config(value = 3)

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                # Visualization of the results of detection.
                result_image = vis_util.visualize_boxes_and_labels_on_image_array(
                    raw_image,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    self.category_index,
                    instance_masks = output_dict.get('detection_masks'),
                    use_normalized_coordinates = False,
                    line_thickness = 2)
                self.terminal_print("INFO", "Detection complete.")
                result_image_name = "{}_[{}]_Result.jpg".format(ts.strftime("%d-%m-%y"), self.counter)
                result_image_path = os.path.join(self.SAVING_DIR, result_image_name)
                cv2.imwrite(result_image_path, result_image)
                self.terminal_print("SAVED", result_image_name)
                self.show_result(result_image_path)

    def show_result(self,path):
        """creates a window that displays the image result"""
        # add a progress on the progress bar
        self.progressBar.config(value = 4)
        image_open = Image.open(path)
        result = ImageTk.PhotoImage(image_open)
        self.win = Toplevel()
        self.win.title("RESULT")
        self.win.attributes('-topmost', True)
        self.win.protocol("WM_DELETE_WINDOW", self.close_result)
        self.win.geometry(f"+{self.root.winfo_x()-5}+{self.root.winfo_y()+23}")
        panel = Label(self.win, image = result); panel.image = result
        panel.pack()
        self.detectionButton.config(state = NORMAL)
    
    def close_result(self):
        self.progressBar.config(value = 0)
        self.win.destroy()

    def change_saving_dir(self):
        """change the path where the captured and relust images will be save"""
        self.SAVING_DIR = filedialog.askdirectory()
        self.terminal_print("PATH", f" {self.SAVING_DIR}")

    def change_model(self):
        """change the path to trained model(.pb file)"""
        path_to_frozen_graph = filedialog.askopenfilename(
            title = "Select Trained Model File(.pb)", filetypes=[("Model File", "*.pb")])
        if not path_to_frozen_graph:
            path_to_frozen_graph = PATH_TO_FROZEN_GRAPH 
        self.terminal_print("PATH", f" {path_to_frozen_graph}")
        self.load_inference_graph(path_to_frozen_graph)

    def change_label(self):
        """change the path to labels(.pbtxt file)"""
        path_to_labels = filedialog.askopenfilename(
            title = "Select Label File(.pbtxt)", filetypes=[("Label File", "*.pbtxt")])
        if not path_to_labels:
            path_to_labels = PATH_TO_LABELS
        self.terminal_print("PATH", f" {path_to_labels}")
        self.load_label(path_to_labels)

    def load_defaults(self):
        """calls the default start up settings and load it to memory"""
        self.terminal_print("INFO", "Load default settings")
        t1 = threading.Thread(target = self.load_inference_graph, args = (PATH_TO_FROZEN_GRAPH,)).start() # run process in seperate thread to avoid mainloop freeze
        t2 = threading.Thread(target = self.load_label, args = (PATH_TO_LABELS,)).start() # run process in seperate thread to avoid mainloop freeze


if __name__ == "__main__":
    # start the app
    run = Application()
    run.root.mainloop()
