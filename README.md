# SIDE-security_cam_v2
An upgrade of the security_cam repo.

This is an upgrade to [SIDE-security_cam repo](https://github.com/GianFederico/SIDE-security_cam). 
This version does everything that the previous versione did, but it also recognizes the faces in the frame using haarcascade to detect the face and LBPH to recognize it.
At the moment it only recognizes me, but you can add any folder containing several pictures of the person you want the program to identificate.

Plus: There is also an attempt to straighten roi (used for the recognition) through the position of the eyes.

Plus2: You can choose which OpenCV recognition methods between LBPH, Fisherfaces and Eigenfaces and just use the one that suits your data the best.
In order to choose a different method to have open the face-train.py and remove the comments about the one you interested in and retrain the model. Then remove the comments also in camera.py.
![photo_2022-05-23_19-03-59](https://user-images.githubusercontent.com/48125720/189713457-49bdd625-eac8-405b-aa60-6a90fd3402de.jpg)
