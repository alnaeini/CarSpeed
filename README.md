# DashCamp_CarSpeed

This repos is an solution to the commaai challenge. that challenge to predict the car speed with only video from Dash Cam located under mirror in front. 

Check this link for more details about the challenge and downloading the data: [speedchallenge](https://github.com/commaai/speedchallenge).

Based on previous attempts, one of the fastest approach in terms of training and performance is to apply the **Optical flow**.

Optical flow is one of the most basic concepts in computer vision, and refers to the apparent motion of objects in the image due to the relative motion between the camera and the scene.

Using Optical flow, can calculate the two components of speed(u,v) using the following equations: 

![equation](OpticalFlowEquation.png)




Split up the data into train(95%) and validation(5%). 
<br>
MSE -
 - Train - 4.7
 - Validation - 2.66
