## Flex-Defect-Detection

This is a project that utilize opensource machine learning and machine vision tools(Google's Tensorflow with OpenCV ) to assist in identifying cosmetic defects for electronic products.

Before                     |  After
:-------------------------:|:-------------------------:
<img src="https://github.com/jericovalino/Flex-Defect-Detection/blob/master/assets/detection.PNG" alt="GUI" width="430"/>  |  <img src="https://github.com/jericovalino/Flex-Defect-Detection/blob/master/assets/detection1.PNG" alt="GUI" width="430"/>



#### The GUI of the Application
To create the graphical user interface, I've used Tkinter, the standard Python interface to the Tk GUI toolkit, and is Python's de facto standard GUI. Tkinter is included with standard Linux, Microsoft Windows and Mac OS X installs of Python.
It uses widgets to create objects such as buttons, labels, frames and etc. To learn more about Tkinter and how to start creating simple GUI's with it, you can go to this [tkinter documentaion](https://docs.python.org/3/library/tk.html).

#### The Custom Trained M-RCNN Model to Detect Electronic Defects
To train the model, you can follow the instructions from my previous [github repository](https://github.com/jericovalino/Train_Mask_RCNN).
Only 7 defective samples from FPCA(flex printed circuit assembly) Toshiba are used as a basis of the machine learning of what is failed. Then I took more than a couple of pictures of each samples with different angles/orientations. All in all, I've collected 23 images as a data to be used in training the model. <br/>
Here are the image datasets I've used to train the model. 

<img src="https://github.com/jericovalino/Flex-Defect-Detection/blob/master/assets/images.PNG" alt="GUI" width="450"/>

Your can download the label and trained model [here](https://drive.google.com/open?id=10P9eOnwiRWgtlqGMUap13Fp6I2S0Rr3g).

#### Running and Testing the Application
Grab the app.py from this repo and place it inside the C:\Train_Mask_RCNN\Tensorflow_API-Custom_Mask_RCNN folder(it maybe different in yours). <br/>
Open the script using code editor, then check or edit the default "PATHS" to match your set-up.
