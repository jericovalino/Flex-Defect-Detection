from tkinter import filedialog
from tkinter import *
 
root = Tk()
root.filename =  filedialog.askdirectory()
print (root.filename)