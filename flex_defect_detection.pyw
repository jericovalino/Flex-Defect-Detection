"""
Naming Style Guide:
  variable,my_variable : variable
  function,my_function : function
  widget,myWidget      : tkinter widgets
  CONSTANT,MY_CONSTANT : constant
"""

# A bunch of imports
import tensorflow as tf
import cv2
import numpy as np
from tkinter import filedialog, messagebox, ttk
from tkinter import *
from PIL import Image, ImageTk, ImageColor, ImageDraw, ImageFont
from datetime import datetime
import threading
import time
import os
import sys
import abc
import collections
import functools
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import six
# This is needed, for us to import utils since the utils is inside the object_detection folder.
sys.path.append("../models/research")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

CWD_PATH = os.getcwd()                                                                  # Gets the current working directory.
# This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, "IG", "frozen_inference_graph.pb")        # default(CONSTANT). <<< you can edit this path.
path_to_frozen_graph = PATH_TO_FROZEN_GRAPH                                             # may change in runtime.
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, "dataset", "label.pbtxt")                       # default(CONSTANT). <<< you can edit this path.
path_to_labels = PATH_TO_LABELS                                                         # may change in runtime.
# Path to where the capture images will be save.
PATH_TO_SAVE = os.path.join(CWD_PATH, 'save_images')                                    # default(CONSTANT). <<< you can edit this path.
path_to_save = PATH_TO_SAVE # may change in runtime.

NUM_CLASSES = 1                                                     # Number of classes
result_image = None                                                 # Declares a variable that will later store the result image.


class Application:
    def __init__(self):
        """Initialize application which uses OpenCV + Tkinter"""
        self.cam = cv2.VideoCapture(0)                              # capture video frames, 0 is your default video camera.
        self.current_frame = None                                   # current image from the camera.
        self.root = Tk()                                            # initialize root window.
        self.root.overrideredirect(True)                            # removes os default window border.
        self.root.attributes('-topmost', True)                      # makes the window always ontop.

        # Gets both half the screen width/height and window width/height.
        # This coordinates will position the window in the center of the screen. 
        window_position_x = int(self.root.winfo_screenwidth()/2 - (848/2))
        window_position_y = int(self.root.winfo_screenheight()/2 - (665/2))
        
        self.root.geometry(f"848x665+{window_position_x}+{window_position_y}")          # window position(center).
        self.root.iconbitmap(default=os.path.join(CWD_PATH, "assets", "icon.ico"))      # set icon.
        self.root.title("Flex Defect Detection")                                        # set window title.
        self.root.configure(bg='#222')                                                  # root window color.

        # Added bindings to pass windows status to function. This ensures to always remove OS default window atribbutes(e.g. topbar, borders)
        self.root.bind('<Map>', self.check_map)                     # calls a function when mian(root) window is minimized
        self.root.bind('<Unmap>', self.check_map)                   # calls a function when main(root) window is unminimized

        self.is_on = True                                           # variable for toggle switch(for save images) value is 'on' by default.

        # Changes the default style of the progressbar widget for toggle switch.
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.s.layout('blue.Horizontal.TProgressbar', 
                      [('Horizontal.Progressbar.trough',{'children': [('Horizontal.Progressbar.pbar',
                          {'side': 'left', 'sticky': 'w'})],'sticky': 'nswe'}), 
                       ('Horizontal.Progressbar.label', {'sticky': 'w'})]
                     )
        self.s.configure('blue.Horizontal.TProgressbar', text="  on", foreground='#4285F4', background='#4285F4')

        # Declairing variables for GUI.
        self.topBar = Frame(self.root)                                                                  # this is where the red, yellow and green button are placed. 
        blueLine = Frame(self.root,bg='#4285F4',height=7)                                               # the blue line at the top.
        titleBar = Frame(self.root,bg='#222',height=30)                                                 # this is where the titleLabel is placed.
        centerFrame = Frame(self.root,bg='#222')                                                        # this is where the video panel and menuFrame are placed.
        logsFrame = Frame(self.root,bg='#222',height=100)                                               # this is where the logsListBox and LogsScrollBar are placed.
        logsScrollBar = Scrollbar(logsFrame)                                                            # the scrollbar -_-
        self.logsListBox = Listbox(logsFrame,bg="#1c313a",fg="#fff",width=103,height=8,borderwidth=0,   # this is where logs is appearing.
            highlightthickness=0,font=('verdana', 9),yscrollcommand=logsScrollBar.set)
        self.logsListBox.bindtags("all")
        logsScrollBar.config(command=self.logsListBox.yview)
        self.panel = Label(centerFrame,width=91,bg="#1c313a",fg="#fff",text="No signal...")             # this is where the video from the camera appears.
        menuFrame = LabelFrame(centerFrame,text="MENU",bg="#eee")                                       # this is where all the menu buttons are placed.
        logo_img = PhotoImage(file=os.path.join(CWD_PATH, "assets", "logo.PNG"))
        logo = Label(titleBar,image=logo_img,bd=0,bg='#222'); logo.image=logo_img                       # this is the tensorflow logo. Placed inside the titleBar

        # Declaring variables for top bar buttons "red = close", "yellow = minimize", "green = clear logs".
        exit_img = PhotoImage(file=os.path.join(CWD_PATH, "assets", "red.PNG"))                         # variable that stores image "red.PNG"
        min_img = PhotoImage(file=os.path.join(CWD_PATH, "assets", "yellow.PNG"))                       # variable that stores image "yellow.PNG"
        max_img = PhotoImage(file=os.path.join(CWD_PATH, "assets", "green.PNG"))                        # variable that stores image "green.PNG"
        exitButton = Button(self.topBar,image=exit_img,bd=0,command=self.close_window)                  # creates a tkinter widget button
        exitButton.image=exit_img                                                                       # keep a reference! to avoid being deleted by garbage-collector
        maxButton = Button(self.topBar,image=max_img,bd=0,command=self.clear_logs)
        maxButton.image=max_img
        minButton = Button(self.topBar,image=min_img,bd=0,command=self.minimize)
        minButton.image=min_img

        # Added bindings that listens, and calls functions when the topbar is being drag.
        self.topBar.bind('<ButtonPress-1>', self.start_move)        # calls a function when topBar is pressed using mouse left button.
        self.topBar.bind('<ButtonRelease-1>', self.stop_move)       # calls a function when the mouse left button is unpressed.
        self.topBar.bind('<B1-Motion>', self.on_motion)             # calls a function when mouse is moving while topBar is being pressed.

        # Declaring a Label variable that holds the title text.
        titleLabel = Label(titleBar,text="Cosmetic Quality Defect Detection in Electronics",
            bd=0,bg='#222',fg='#fefefe',font=('verdana', 10))

        # Placing the tkinter widgets using grid.
        self.topBar.grid(sticky=(W,E))
        exitButton.grid(column=0,row=0,padx=(3,0))
        minButton.grid(column=1,row=0,padx=(1,0))
        maxButton.grid(column=2,row=0,padx=(1,0))     
        blueLine.grid(row=1,column=0,sticky=(W,E))
        titleBar.grid(row=2,column=0,sticky=(W,E))
        logo.grid(row=0,column=0,pady=3)
        titleLabel.grid(row=0,column=1,padx=10)
        centerFrame.grid(row=3,column=0,sticky=(W,E))
        self.panel.grid(row=0,column=0,sticky=(N,S,W),padx=(3,0))
        menuFrame.grid(row=0,column=1,sticky=(N,E,S,W),padx=(0,3),pady=2)
        logsFrame.grid(row=4,column=0,sticky=(W,E))
        self.logsListBox.grid(row=0,column=0,sticky=(N,E,S,W),padx=(3,0),pady=3)
        logsScrollBar.grid(row=0,column=1,sticky=(N,S,E),padx=(1,3),pady=3)

        # These widgets bellow are placed inside the menuFrame
        # Create a progress bar for detection process.
        self.progressBar = ttk.Progressbar(menuFrame,style='',mode='determinate',
            orient=HORIZONTAL,maximum=4,value=0)
        self.progressBar.grid(column=0,row=0,columnspan=2,padx=(0,1),sticky=(E,W))

        # Creates a button, that when pressed, will take the current frame and then run the detection.
        self.detectionButton = Button(menuFrame,text="RUN DETECTION",bg='#e0e0e0',
            width=23,height=2,relief=GROOVE,font=('verdana', 10),command=self.take_snapshot)
        self.detectionButton.grid(column=0,row=1,columnspan=2,sticky=(E,W))

        # This widget is where the text result appears
        self.resultLabel = Label(menuFrame,text="",height=1,bd=0,bg='#eee',
            fg='green',font=('verdana', 25))
        self.resultLabel.grid(column=0,row=2,columnspan=2,sticky=(E,W),pady=(100))

        # Toggle switch label.
        Label(menuFrame,text="save images").grid(column=0,row=3,columnspan=1,sticky=E)

        # Creates a custom toggle switch using undeterminate progressbar
        self.toggleSwitch = ttk.Progressbar(menuFrame,style='blue.Horizontal.TProgressbar',
            orient='horizontal',length=64,mode='indeterminate',maximum=10,value=10)
        self.toggleSwitch.grid(column=1,row=3,columnspan=1,padx=(0,1),sticky=E)
        # calls a fuction when this widget is pressed.
        self.toggleSwitch.bind('<ButtonPress-1>', self.toggle_switch)

        # Creates a button, that when pressed, will open a fileDialog for user to select saving dir.
        changeDirButton = Button(menuFrame,text="change saving directory",
            relief=GROOVE,command=(self.change_saving_dir))
        changeDirButton.grid(column=0,row=4,columnspan=2,sticky=(E,W))

        # Creates a button, that when pressed, will open a fileDialog for user to select saving dir.
        saveLogsButton = Button(menuFrame,text="export logs as text file",
            relief=GROOVE,command=(self.save_logs))
        saveLogsButton.grid(column=0,row=5,columnspan=2,sticky=(E,W))

        # Creates a button, that when pressed, will call the funtion self.change_model.
        selectModelButton = Button(menuFrame,text="change trained model",
            relief=GROOVE,command=(self.change_model))
        selectModelButton.grid(column=0,row=6,columnspan=2,sticky=(E,W))

        # Creates a button, that when pressed, will call the funtion self.change_label.
        selectLabelButton = Button(menuFrame,text="change label.pbtxt file",
            relief=GROOVE,command=(self.change_label))
        selectLabelButton.grid(column=0,row=7,columnspan=2,sticky=(E,W))

        # Creates a button, that when pressed, will call the funtion self.load_defaults.
        self.restoreDefaultButton = Button(menuFrame,text="restore default settings",
            relief=GROOVE,command=(self.load_defaults))
        self.restoreDefaultButton.grid(column=0,row=8,columnspan=2,sticky=(E,W))

        self.counter = 0                                            # counter for captured image numbering.

        self.video_loop()                                           # calls self.video_loop at start up.
        self.load_defaults()                                        # loads default settings at start-up.

    def check_map(self, event):
        """applies override on deiconify"""
        if str(event) == '<Map event>':
            self.root.overrideredirect(True)

    def start_move(self, event):
        """assigning x and y with the value of x and y coordinates of the cursor when the mouse left button is pressed"""
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        """resets x and y value when the mouse left button is released"""
        self.x = None
        self.y = None

    def on_motion(self, event):
        """rapidly changes the position of the window/s based on the changes in x and y when the top bar is being drag"""
        # you can print the value of x and y to see the value changes in in real-time.
        x = (event.x_root - self.x - self.topBar.winfo_rootx() + self.topBar.winfo_rootx())             # calculates the changes in x
        y = (event.y_root - self.y - self.topBar.winfo_rooty() + self.topBar.winfo_rooty())             # calculates the changes in y
        self.root.geometry(f"+{x}+{y}")                             # this sets the x and y position of the main window.
        try:                                                        # if result window is present: also changes its position.
            self.win.attributes('-topmost',True)
            self.win.geometry(f"+{x-5}+{y+23}")
        except:                                                     # else if result window is not present: do nothing
            pass

    def video_loop(self):
        """get frame from the video stream and show it in panel every 30 milliseconds"""
        self.ok, self.frame = self.cam.read()                                           # read frame from video stream.
        if self.ok:                                                                     # if frame is captured without any errors.
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)                     # convert colors from BGR to RGBA.
            self.current_frame = Image.fromarray(cv2image)                              # convert image for PIL.
            frametk = ImageTk.PhotoImage(image=self.current_frame)                      # convert image for tkinter.
            self.panel.frametk = frametk                                                # reference frametk so it does not be deleted by garbage-collector.
            self.panel.config(image=frametk, width=640, height=480)                     # show the image in panel.
        if not self.ok:                                                                 # if not: establish the camera again and wait for the camera to be ok. 
            frametk = None                                                              # empty the variable that holds the frame image
            self.panel.frametk = None                                                   # empty the reference
            self.cam = cv2.VideoCapture(0)                                              # tries again to establish the camera
        self.root.after(30, self.video_loop)                                            # call the same function after 30 milliseconds.

    def print_logs(self, prompt, message):
        """print something in the logs(logsListBox)"""
        time = datetime.now()
        self.logsListBox.insert(END, f"[{time.hour}:{time.minute}:{time.second}] [{prompt}] {message}")
        self.logsListBox.see(END)

    def clear_logs(self):
        """clears the logs(logsListBox)"""
        self.logsListBox.delete(0, END)

    def save_logs(self):
        logs = list(self.logsListBox.get(0,END))
        logs = [row + '\n' for row in logs]
        filename = filedialog.asksaveasfile("w", title="Save Logs(.txt)", defaultextension=".txt")
        if not filename:
            pass
        else:
            filename.writelines(logs)
            filename.close()
            self.clear_logs()
            self.print_logs("SAVED",f"{filename.name}")

    def close_window(self):
        """Destroy the window and release all resources """
        is_yes = messagebox.askyesno("Exit program",
            "You are about to close the program. Are you sure you want to continue?")
        if is_yes:
            self.root.destroy()                                     # destroys the main window.
            self.cam.release()                                      # releases the camera.

    def minimize(self):
        """minimize the window"""
        self.root.wm_withdraw()
        self.root.wm_overrideredirect(False)
        self.root.wm_iconify()

    def toggle_switch(self, event):
        """toggle switch(on/off) save captured and result images"""
        if self.is_on:                                              # if switch is on: turn off.
            def turn_off():
                i = 10
                while i >= 0:                                       # this loop does the toggle slide to left animation.
                    self.toggleSwitch['value'] = i
                    time.sleep(0.01)
                    i -= 1
                self.s.layout('blue.Horizontal.TProgressbar', 
                            [('Horizontal.Progressbar.trough',{'children': [('Horizontal.Progressbar.pbar',
                                {'side': 'left', 'sticky': 'w'})],'sticky': 'nswe'}), 
                            ('Horizontal.Progressbar.label', {'sticky': 'e'})]
                            )
                self.s.configure('blue.Horizontal.TProgressbar', text="off   ", foreground='grey', background='grey')
            threading.Thread(target=turn_off).start()               # runs the animation process in seperate thread to avoid mainloop feeze.
            self.is_on =False
        else:                                                       # if switch is off: turn on.
            def turn_on():
                i = 0
                while i <= 10:                                      # this loop does the toggle slide to right animation.
                    self.toggleSwitch['value'] = i
                    time.sleep(0.01)
                    i += 1
                self.s.layout('blue.Horizontal.TProgressbar', 
                            [('Horizontal.Progressbar.trough',{'children': [('Horizontal.Progressbar.pbar',
                                {'side': 'left', 'sticky': 'w'})],'sticky': 'nswe'}), 
                            ('Horizontal.Progressbar.label', {'sticky': 'w'})]
                            )
                self.s.configure('blue.Horizontal.TProgressbar', text="  on", foreground='#4285F4', background='#4285F4')
            threading.Thread(target=turn_on).start()                # runs the animation process in seperate thread to avoid mainloop feeze.
            self.is_on =True
        if self.is_on:
            self.print_logs("SAVE", "Turned ON")
        else:
            self.print_logs("SAVE", "Turned OFF")

    def load_defaults(self):
        """calls the default start up settings and load it to memory"""
        self.detectionButton.config(state=DISABLED)
        self.restoreDefaultButton.config(state=DISABLED)
        self.print_logs("INFO", "Loading default settings")
        path_to_save = PATH_TO_SAVE
        self.print_logs("SAVING DIRECTORY", f"{path_to_save}")
        t1 = threading.Thread(target = self.load_inference_graph, args = (PATH_TO_FROZEN_GRAPH,)).start()   # run process in seperate thread to avoid mainloop freeze
        t2 = threading.Thread(target = self.load_label, args = (PATH_TO_LABELS,)).start()                   # run process in seperate thread to avoid mainloop freeze

    def change_saving_dir(self):
        """change the path where the captured and relust images will be save"""
        global path_to_save
        path = filedialog.askdirectory()
        if not path:
            pass
        else:
            path_to_save = path
            self.print_logs("SAVING DIRECTORY", f"{path_to_save}")

    def change_model(self):
        """change the path to trained model(.pb file)"""
        path = filedialog.askopenfilename(
            title = "Select Trained Model File(.pb)", filetypes=[("Model File", "*.pb")])
        if not path:
            pass
        else:
            path_to_frozen_graph = path
            self.print_logs("SELECTED", f"{path_to_frozen_graph}")
            self.load_inference_graph(path_to_frozen_graph)

    def load_inference_graph(self,path_to_frozen_graph):
        """load the (.pb) file into memory"""
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.print_logs("INFO", "Loading the Inference Graph into memory. Done")
        self.restoreDefaultButton.config(state=NORMAL)
        self.detectionButton.config(state=NORMAL)

    def change_label(self):
        """change the path to labels(.pbtxt file)"""
        path = filedialog.askopenfilename(
            title = "Select Label File(.pbtxt)", filetypes=[("Label File", "*.pbtxt")])
        if not path:
            pass
        else:
            path_to_labels = path
            self.print_logs("SELECTED", f"{path_to_labels}")
            self.load_label(path_to_labels)

    def load_label(self,path_to_labels):
        """load the (.pbtxt) file into memory"""
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.print_logs("INFO", "Loading labels into memory. Done.")

    def take_snapshot(self):
        """Take snapshot, then load the captured image to inference"""
        if not self.ok:                                                                                 # if camera is not ok: print warning.
            self.print_logs("WARNING", "No connected camera found!!!")            
        else:                                                                                           # else: process the captured image.
            self.counter += 1
            self.detectionButton.config(state = DISABLED)
            result_text = None
            try:                                                                                        # if the result window is present: close the result window.
                self.close_result()
            except:
                pass
            if self.counter == 1:                                                                       # print some information in the first run detection.
                self.clear_logs()
                self.print_logs("INFO", "This is the first run. This might take up to 20 seconds...")
            self.print_logs(f"{self.counter}", f"{'-' * 110}")
            if self.is_on:                                                                              # if toggle switch is on: this saves the captured image
                ts = datetime.now()
                captured_image_name = "{}_[{}].jpg".format(ts.strftime("%d-%m-%y"),self.counter)        # construct filename
                captured_image_path = os.path.join(path_to_save, captured_image_name)                   # construct output path
                self.print_logs("SAVED", captured_image_name)
                cv2.imwrite(captured_image_path, self.frame)                                            # write/save the image

            imageopen = Image.fromarray(self.frame)
            image_np = self.load_image_into_numpy_array(imageopen)
            t1 = threading.Thread(target=self.run_inference, args=(self.frame,image_np)).start()        # run inference in seperate thread to avoid mainloop freeze
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
                self.print_logs("INFO", "Evaluating captured image. Please wait...")                
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

                self.progressBar.config(value = 2)
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict = {image_tensor: np.expand_dims(image, 0)})

                self.progressBar.config(value = 3)

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                
                global result_image
                # Visualization of the results of detection.
                result_image = visualize_boxes_and_labels_on_image_array(
                    raw_image,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    self.category_index,
                    instance_masks = output_dict.get('detection_masks'),
                    use_normalized_coordinates = True,
                    line_thickness = 1)
                self.print_logs("INFO", "Detection complete.")
                if self.is_on:
                    ts = datetime.now()
                    if result_text == "FAIL":
                        result_image_name = "{}_[{}]_FAIL.jpg".format(ts.strftime("%d-%m-%y"), self.counter)
                    else:
                        result_image_name = "{}_[{}]_PASS.jpg".format(ts.strftime("%d-%m-%y"), self.counter)
                    result_image_path = os.path.join(path_to_save, result_image_name)
                    cv2.imwrite(result_image_path, result_image)
                    self.print_logs("SAVED", result_image_name)
                self.show_result()

    def show_result(self):
        """creates a window that displays the image result"""
        self.progressBar.config(value = 4)
        self.win = Toplevel()
        self.win.title("RESULT")
        self.win.attributes('-topmost', True)
        self.win.protocol("WM_DELETE_WINDOW", self.close_result)
        self.win.geometry(f"+{self.root.winfo_x()-5}+{self.root.winfo_y()+23}")
        result_panel = Label(self.win)
        cv2img = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGBA)                                         # convert colors from BGR to RGBA.
        cv2img = Image.fromarray(cv2img)                                                                # convert image for PIL.
        cv2img = ImageTk.PhotoImage(image=cv2img)                                                       # convert image for tkinter.
        result_panel.cv2img = cv2img                                                                    # anchor cv2img so it does not be deleted by garbage-collector.
        result_panel.config(image = cv2img, width = 640, height = 480)                                  # show the image in panel.
        result_panel.pack()
        self.detectionButton.config(state = NORMAL)
        if result_text == "FAIL":
            self.resultLabel.config(text="FAIL",fg="red")
        else:
            self.resultLabel.config(text="PASS",fg="green")

    def close_result(self):
        """closes the result window, removing resultLabel and resets the progress bar"""
        self.progressBar.config(value = 0)
        self.resultLabel.config(text="")
        self.win.destroy()


""" ========= The codes bellow is just a modified visualization_utils.py ===========
A set of functions that are used for visualization.
These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself."""
def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color='red', thickness=4,
                                     display_str_list=(), use_normalized_coordinates=True):

    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4,
                               display_str_list=(), use_normalized_coordinates=True):

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image, boxes, color='red', thickness=4, display_str_list_list=()):
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image, boxes, color='red', thickness=4, display_str_list_list=()):

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness, display_str_list)


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        keypoints=keypoints,
        **kwargs)


def _visualize_boxes_and_masks_and_keypoints(
        image, boxes, classes, scores, masks, keypoints, category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        keypoints=keypoints,
        **kwargs)


def _resize_original_image(image, image_shape):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_images(
        image,
        image_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=True)
    return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index, original_image_spatial_shape=None,
                                         true_image_shape=None, instance_masks=None, keypoints=None, max_boxes_to_draw=20,
                                         min_score_thresh=0.2, use_normalized_coordinates=True):

    # Additional channels are being ignored.
    if images.shape[3] > 3:
        images = images[:, :, :, 0:3]
    elif images.shape[3] == 1:
        images = tf.image.grayscale_to_rgb(images)
    visualization_keyword_args = {
        'use_normalized_coordinates': use_normalized_coordinates,
        'max_boxes_to_draw': max_boxes_to_draw,
        'min_score_thresh': min_score_thresh,
        'agnostic_mode': False,
        'line_thickness': 4
    }
    if true_image_shape is None:
        true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
    else:
        true_shapes = true_image_shape
    if original_image_spatial_shape is None:
        original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
    else:
        original_shapes = original_image_spatial_shape

    if instance_masks is not None and keypoints is None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_masks,
            category_index=category_index,
            **visualization_keyword_args)
        elems = [
            true_shapes, original_shapes, images, boxes, classes, scores,
            instance_masks
        ]
    elif instance_masks is None and keypoints is not None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_keypoints,
            category_index=category_index,
            **visualization_keyword_args)
        elems = [
            true_shapes, original_shapes, images, boxes, classes, scores, keypoints
        ]
    elif instance_masks is not None and keypoints is not None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_masks_and_keypoints,
            category_index=category_index,
            **visualization_keyword_args)
        elems = [
            true_shapes, original_shapes, images, boxes, classes, scores,
            instance_masks, keypoints
        ]
    else:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes,
            category_index=category_index,
            **visualization_keyword_args)
        elems = [
            true_shapes, original_shapes, images, boxes, classes, scores
        ]

    def draw_boxes(image_and_detections):
        """Draws boxes on image."""
        true_shape = image_and_detections[0]
        original_shape = image_and_detections[1]
        if true_image_shape is not None:
            image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                               [true_shape[0], true_shape[1], 3])
        if original_image_spatial_shape is not None:
            image_and_detections[2] = _resize_original_image(
                image, original_shape)

        image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                      tf.uint8)
        return image_with_boxes

    images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
    return images


def draw_side_by_side_evaluation_image(eval_dict, category_index, max_boxes_to_draw=20,
                                       min_score_thresh=0.2, use_normalized_coordinates=True):

    detection_fields = fields.DetectionResultFields()
    input_data_fields = fields.InputDataFields()

    images_with_detections_list = []

    # Add the batch dimension if the eval_dict is for single example.
    if len(eval_dict[detection_fields.detection_classes].shape) == 1:
        for key in eval_dict:
            if key != input_data_fields.original_image:
                eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

    for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
        instance_masks = None
        if detection_fields.detection_masks in eval_dict:
            instance_masks = tf.cast(
                tf.expand_dims(
                    eval_dict[detection_fields.detection_masks][indx], axis=0),
                tf.uint8)
        keypoints = None
        if detection_fields.detection_keypoints in eval_dict:
            keypoints = tf.expand_dims(
                eval_dict[detection_fields.detection_keypoints][indx], axis=0)
        groundtruth_instance_masks = None
        if input_data_fields.groundtruth_instance_masks in eval_dict:
            groundtruth_instance_masks = tf.cast(
                tf.expand_dims(
                    eval_dict[input_data_fields.groundtruth_instance_masks][indx],
                    axis=0), tf.uint8)

        images_with_detections = draw_bounding_boxes_on_image_tensors(
            tf.expand_dims(
                eval_dict[input_data_fields.original_image][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_boxes][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_classes][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_scores][indx], axis=0),
            category_index,
            original_image_spatial_shape=tf.expand_dims(
                eval_dict[input_data_fields.original_image_spatial_shape][indx],
                axis=0),
            true_image_shape=tf.expand_dims(
                eval_dict[input_data_fields.true_image_shape][indx], axis=0),
            instance_masks=instance_masks,
            keypoints=keypoints,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            use_normalized_coordinates=use_normalized_coordinates)
        images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
            tf.expand_dims(
                eval_dict[input_data_fields.original_image][indx], axis=0),
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
            tf.expand_dims(
                tf.ones_like(
                    eval_dict[input_data_fields.groundtruth_classes][indx],
                    dtype=tf.float32),
                axis=0),
            category_index,
            original_image_spatial_shape=tf.expand_dims(
                eval_dict[input_data_fields.original_image_spatial_shape][indx],
                axis=0),
            true_image_shape=tf.expand_dims(
                eval_dict[input_data_fields.true_image_shape][indx], axis=0),
            instance_masks=groundtruth_instance_masks,
            keypoints=None,
            max_boxes_to_draw=None,
            min_score_thresh=0.0,
            use_normalized_coordinates=use_normalized_coordinates)
        images_with_detections_list.append(
            tf.concat([images_with_detections, images_with_groundtruth], axis=2))
    return images_with_detections_list


def draw_keypoints_on_image_array(image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                         'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
        np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index, instance_masks=None,
                                              instance_boundaries=None, keypoints=None, use_normalized_coordinates=False,
                                              max_boxes_to_draw=20, min_score_thresh=.5, agnostic_mode=False, line_thickness=4,
                                              groundtruth_box_visualization_color='black', skip_scores=False, skip_labels=False):

    global result_text
    result_text = None
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                result_text = "FAIL"
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100*scores[i]))
                    else:
                        display_str = ''#'{}: {}%'.format(
                            #display_str, int(100*scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = 'Blue'

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                image,
                box_to_instance_masks_map[box],
                color=color
            )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image,
                box_to_instance_boundaries_map[box],
                color='red',
                alpha=1.0
            )
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates)
    return image


def add_cdf_image_summary(values, name):
    def cdf_plot(values):
        """Numpy function to plot CDF."""
        normalized_values = values / np.sum(values)
        sorted_values = np.sort(normalized_values)
        cumulative_values = np.cumsum(sorted_values)
        fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                                / cumulative_values.size)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        ax.plot(fraction_of_examples, cumulative_values)
        ax.set_ylabel('cumulative normalized values')
        ax.set_xlabel('fraction of examples')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
        return image
    cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
    tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
    def hist_plot(values, bins):
        """Numpy function to plot hist."""
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        y, x = np.histogram(values, bins=bins)
        ax.plot(x[:-1], y)
        ax.set_ylabel('count')
        ax.set_xlabel('value')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(
            fig.canvas.tostring_rgb(), dtype='uint8').reshape(
                1, int(height), int(width), 3)
        return image
    hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
    tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, category_index, max_examples_to_draw=5, max_boxes_to_draw=20,
                 min_score_thresh=0.2, use_normalized_coordinates=True, summary_name_prefix='evaluation_image'):

        self._category_index = category_index
        self._max_examples_to_draw = max_examples_to_draw
        self._max_boxes_to_draw = max_boxes_to_draw
        self._min_score_thresh = min_score_thresh
        self._use_normalized_coordinates = use_normalized_coordinates
        self._summary_name_prefix = summary_name_prefix
        self._images = []

    def clear(self):
        self._images = []

    def add_images(self, images):
        """Store a list of images, each with shape [1, H, W, C]."""
        if len(self._images) >= self._max_examples_to_draw:
            return

        # Store images and clip list if necessary.
        self._images.extend(images)
        if len(self._images) > self._max_examples_to_draw:
            self._images[self._max_examples_to_draw:] = []

    def get_estimator_eval_metric_ops(self, eval_dict):
        if self._max_examples_to_draw == 0:
            return {}
        images = self.images_from_evaluation_dict(eval_dict)

        def get_images():
            """Returns a list of images, padded to self._max_images_to_draw."""
            images = self._images
            while len(images) < self._max_examples_to_draw:
                images.append(np.array(0, dtype=np.uint8))
            self.clear()
            return images

        def image_summary_or_default_string(summary_name, image):
            """Returns image summaries for non-padded elements."""
            return tf.cond(
                tf.equal(tf.size(tf.shape(image)), 4),
                lambda: tf.summary.image(summary_name, image),
                lambda: tf.constant(''))

        update_op = tf.py_func(self.add_images, [[images[0]]], [])
        image_tensors = tf.py_func(
            get_images, [], [tf.uint8] * self._max_examples_to_draw)
        eval_metric_ops = {}
        for i, image in enumerate(image_tensors):
            summary_name = self._summary_name_prefix + '/' + str(i)
            value_op = image_summary_or_default_string(summary_name, image)
            eval_metric_ops[summary_name] = (value_op, update_op)
        return eval_metric_ops

    @abc.abstractmethod
    def images_from_evaluation_dict(self, eval_dict):
        raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
    """Class responsible for single-frame object detection visualizations."""

    def __init__(self, category_index, max_examples_to_draw=5, max_boxes_to_draw=20, min_score_thresh=0.2,
                 use_normalized_coordinates=True, summary_name_prefix='Detections_Left_Groundtruth_Right'):

        super(VisualizeSingleFrameDetections, self).__init__(category_index=category_index, max_examples_to_draw=max_examples_to_draw,
                                                             max_boxes_to_draw=max_boxes_to_draw, min_score_thresh=min_score_thresh,
                                                             use_normalized_coordinates=use_normalized_coordinates, summary_name_prefix=summary_name_prefix)

    def images_from_evaluation_dict(self, eval_dict):
        return draw_side_by_side_evaluation_image(
            eval_dict, self._category_index, self._max_boxes_to_draw,
            self._min_score_thresh, self._use_normalized_coordinates)



if __name__ == "__main__":
    # start the app
    run = Application()
    run.root.mainloop()
