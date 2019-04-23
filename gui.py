from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
import numpy

CWD_PATH = os.getcwd()

class Application:

    def __init__(self):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        
        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window
        self.root.title("FLEX DAMAGE DETECTION")  # set window title
        self.root.resizable(False,False)
        self.root.configure(bg="#37474f")
        # self.destructor function gets fired when the window is closed
        self.root.protocol('DELETE_WINDOW', self.destructor)


        self.TopFrame = Frame(self.root, bg = "#102027", width = 800, height = 10)
        self.panel = tk.Label(self.root)  # initialize image panel
        self.LblFrame = LabelFrame(self.root,text="Details",background="#62727b")

        
        self.TopFrame.grid(sticky=N,columnspan=5,pady=(0,30))
        self.panel.grid(row=1,column=0,sticky=W, columnspan=3,padx=(3,0))
        self.LblFrame.grid(row=1,column=3,columnspan=2,sticky=(N,E,S,W),padx=(0,3))
        #self.ttl = Label(self.LblFrame,text="jerico").grid()

        

        self.counter = 0
        # create a button, that when pressed, will take the current frame and save it to file
        btn = tk.Button(self.LblFrame, text="evaluate", command=self.take_snapshot)
        btn.configure(bd=0, background="white")
        btn.grid(column = 0, row = 2, sticky = (E,W))
        # create a button, that when pressed, will take the current frame and save it to file
        self.entr = tk.Entry(self.LblFrame, text="evaluate")
        self.entr.configure(bd=1, background="#9e9e9e")
        self.entr.grid(column = 0, row = 3, sticky = (E,W))

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            frametk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.frametk = frametk  # anchor frametk so it does not be deleted by garbage-collector
            self.panel.config(image=frametk)  # show the image
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
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


if __name__ == "__main__":
    # start the app
    print("[INFO] starting...")
    run = Application()
    run.root.mainloop()
