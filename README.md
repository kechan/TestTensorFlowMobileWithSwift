# TestTensorFlowMobileWithSwift

## Sample project to test Tensorflow Mobile with UIKit and AVFoundation related code re-written in Swift.

This is based on:

   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios
   
provided as example on how to run Inception model on iOS to do Image recognition. Here, I have combined the /simple and /camara into a single test project. Since Tensorflow API is written in C++, an Objective-C wrapper is created and referenced by a bridging header. I have also translated objective-C into Swift (mostly in the viewcontrollers) and make minor modifications to the UI to use Storyboard and Autolayouts. This project may be useful for conveniently sanity test Convolution Neural Nets if you prefer to work with Swift and more modern UI practices.

Tested on Xcode 9.0 with Cocoapod 1.3.1 (but may also work for older versions).

Steps:
1) Clone the project
2) % cd TestTensorFlowMobileWithSwift
3) % pod install
4) % open TestTensorFlowMobileWithSwift.xcworkspace

If you are not able to run, please post them on Issues.
