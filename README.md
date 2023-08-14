# Infantry & Vehicle Detection

## Example GIF from Program Output
![](https://github.com/DYLAB2331/infantry-and-vehicle-detection/blob/main/test.gif)

This is a C++ program that takes in an input video and uses a custom trained VOLOv5l model to detect and track infantry and military vehicles. The model was trained on a set of drone images from the Russo-Ukrainian War and then converted to ONNX format. The dataset was not extensive, and so the model has limitations and will make mistakes fairly frequently and/or not be able to detect objects in some instances. The resulting video will have bounding boxes drawn around an object based on its confidence score and class score. This program was built primarily to obtain familiarity with OpenCV and its Deep Neural Networks module.

## Dependencies

The program depends on OpenCV to be built with CUDA support. This is so that FPS is not extremely limited. A guide can be found [here](https://haroonshakeel.medium.com/build-opencv-4-4-0-with-cuda-gpu-support-on-windows-10-without-tears-aa85d470bcd0).
