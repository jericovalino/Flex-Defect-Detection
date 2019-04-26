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
        minbtn = Button(self.topBar, image=minImg, bd=0, command=self.createWindow); minbtn.image = minImg
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

        titleLabel = Label(self.titleBar, text="Cosmetic Quality Defect Detection", bd=0, bg="#222222", foreground='#fefefe',font=('verdana', 10))

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

#start
