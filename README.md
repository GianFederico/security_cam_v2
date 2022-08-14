# SIDE-security_cam_v2
An upgrade of the security_cam repo.

This is an upgrade to SIDE-security_cam repo. 
This version does everything that the previous versione did, but it also recognizes the faces in the frame using haarcascade to detect the face and LBPH to recognize it.
At the moment it only recognizes me, but you can add any folder containing several pictures of the person you want the program to identificate.

Plus: There is also an attempt to straighten roi (used for the recognition) through the position of the eyes.

Plus2: You can choose which OpenCV recognition methods between LBPH, Fisherfaces and Eigenfaces and just use the one that suits your data the best.
In order to choose a different method to have open the face-train.py and remove the comments about the one you interested in and retrain the model.
