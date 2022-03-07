# gesture-car
A Gesture Controlled Robot Arduino Car

#Approach
1. Use a camera to catch live gesture image input.
2. Transfer the input image to the gesture recognition module.
3. Recognize the gesture and send serial input using RF Transmitter to Arduino.
4. Convert the signal into corresponding voltages to move the motors.

#Contents
gesture-recog.py to train the gestures and obtain the vgg model.
Run the model and transmit signal to the Arduino car using rf_control.py file.

Flow Diagram
![image](https://user-images.githubusercontent.com/18104656/156959111-d1104b93-1597-4302-866c-e83f868a5ab1.png)


