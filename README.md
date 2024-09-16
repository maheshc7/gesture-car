# gesture-car
[A Gesture Controlled Robot Arduino Car](https://docs.google.com/presentation/d/1Lb0c5RhTpcOTH-1fRDtfHK_VzTmxMmPF5R6QQcW8Z_s/pub?start=true&loop=false&delayms=3000)

Authors: Ananya Sundararajan, Mahesh Narpat Chand
## Approach
1. Use a camera to catch live gesture image input.
2. Transfer the input image to the gesture recognition module.
3. Recognize the gesture and send serial input using RF Transmitter to Arduino.
4. Convert the signal into corresponding voltages to move the motors.

##  Contents
1. Run gesture-recog.py to train the gestures and obtain the vgg model.
2. Run the model and transmit signal to the Arduino car using rf_control.py file.

## Flow Diagram
![image](https://user-images.githubusercontent.com/18104656/156959111-d1104b93-1597-4302-866c-e83f868a5ab1.png)
