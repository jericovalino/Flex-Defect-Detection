from tkinter import *
from PIL import Image, ImageTk
#import argparse
import datetime
import cv2
import os
import numpy

CWD_PATH = os.getcwd()

class Application:

    def __init__(self):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.cam = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.current_image = None  # current image from the camera
        self.root = Tk()  # initialize root window
        self.root.overrideredirect(True)

        # Gets both half the screen width/height and window width/height
        screenLengthX = int(self.root.winfo_screenwidth()/2 - (800/2))
        screenLengthY = int(self.root.winfo_screenheight()/2 - (543/2))

        self.root.geometry(f"800x543+{screenLengthX}+{screenLengthY}") # window position(center)

        self.root.title("")  # set window title
        self.root.resizable(False,False)
        self.root.configure(bg="#222222") #bg color
        self.root.protocol('DELETE_WINDOW', self.destructor)


        self.topBar = Frame(self.root)
        self.blueLine = Frame(self.root, bg="#4285F4", width=800, height=7)
        self.panel = Label(self.root)  # initialize image panel
        self.lblFrame = LabelFrame(self.root,text="Flex Defect Detection",background="#f5f5f5")
        self.banner = Label(self.root, text="Flex Defect Detection", bg="#222222", font=("Courier", 12), foreground="#f5f5f5")


        """declaring variables for top bar buttons starts here"""
        exitImg = PhotoImage(file = "red.PNG")
        minImg = PhotoImage(file = "yellow.PNG")
        maxImg = PhotoImage(file = "green.PNG")
        exitbtn = Button(self.topBar, image=exitImg, bd=0, command=self.destructor); exitbtn.image = exitImg
        maxbtn = Button(self.topBar, image=maxImg, bd=0); maxbtn.image = maxImg
        minbtn = Button(self.topBar, image=minImg, bd=0, command=self.createWindow); minbtn.image = minImg
        """declaring variables for top bar buttons ends here"""
        
        
        """initializing dragging feature starts here"""
        self.drag = self.topBar
        self.drag.bind('<ButtonPress-1>', self.StartMove)
        self.drag.bind('<ButtonRelease-1>', self.StopMove)
        self.drag.bind('<B1-Motion>', self.OnMotion)
        """initializing dragging feature ends here"""

 

        self.topBar.grid(columnspan=5)        
        self.blueLine.grid(row=1,column=0,sticky=N,columnspan=5, pady=(0,30))
        self.panel.grid(row=2,column=0,sticky=W, columnspan=3,padx=(3,0))
        self.lblFrame.grid(row=2,column=3,columnspan=2,sticky=(N,E,S,W),padx=(0,3))
        #self.banner.grid(row=0, column=0, sticky=W)
        #self.ttl = Label(self.lblFrame,text="jerico").grid()

        

        self.counter = 0
        # create a button, that when pressed, will take the current frame and save it to file
        detectbtn = Button(self.lblFrame, text="Run Detection", command=self.take_snapshot, height=3)
        detectbtn.configure( background="white", width=20, relief=GROOVE)
        detectbtn.grid(column = 0, row = 2, sticky = (E,W))


        exitbtn.grid(column = 0, row = 0, padx=(3,0))
        minbtn.grid(column = 1, row = 0)
        maxbtn.grid(column = 2, row = 0, padx=(0,742))

        # create a button, that when pressed, will take the current frame and save it to file
        self.entr = Entry(self.lblFrame, text="evaluate")
        self.entr.configure(bd=1, background="#9e9e9e")
        self.entr.grid(column = 0, row = 3, sticky = (E,W))

        #WindowDraggable(self.topBar)


        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    """functions for dragging feature starts here"""
    def StartMove(self, event):
        self.x = event.x
        self.y = event.y

    def StopMove(self, event):
        self.x = None
        self.y = None

    def OnMotion(self,event):
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
        else:
            print("no connected camera!")
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

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

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.cam.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

    def createWindow(self):
        cv2.namedWindow("test")


class WindowDraggable(Application):

    def __init__(self, label):
        #Application.__init__(self)
        self.label = label
        label.bind('<ButtonPress-1>', self.StartMove)
        label.bind('<ButtonRelease-1>', self.StopMove)
        label.bind('<B1-Motion>', self.OnMotion)

    def StartMove(self, event):
        self.x = event.x
        self.y = event.y

    def StopMove(self, event):
        self.x = None
        self.y = None

    def OnMotion(self,event):
        x = (event.x_root - self.x - self.label.winfo_rootx() + self.label.winfo_rootx())
        y = (event.y_root - self.y - self.label.winfo_rooty() + self.label.winfo_rooty())
        #print(x)
        #print(y)
        #Application.root.geometry(f"+{x}+{y}")

if __name__ == "__main__":
    # start the app
    print("[INFO] starting...")
    run = Application()
    run.root.mainloop()
