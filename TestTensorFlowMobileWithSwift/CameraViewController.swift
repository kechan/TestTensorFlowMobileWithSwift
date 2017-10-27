//
//  CameraViewController.swift
//  TestTensorFlowMobileWithSwift
//
//  Created by Kelvin C on 10/21/17.
//  Copyright © 2017 Kelvin Chan. All rights reserved.
//

import UIKit
import AVFoundation

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, TensorflowWrapperDelegate {
    
    // If you have your own model, modify this to the file name, and make sure
    // you've added the file to your app resources too.
    
    static let model_file_name = "tensorflow_inception_graph"
    static let model_file_type = "pb"
    
    let model_uses_memory_mapping = false
    
    static let labels_file_name = "imagenet_comp_graph_label_strings"
    static let labels_file_type = "txt"
    
    let wanted_input_width: Int32 = 224
    let wanted_input_height: Int32 = 224
    let wanted_input_channels: Int32 = 3
    
    let input_mean: Float = 117.0
    let input_std: Float = 1.0
    
//    let input_mean: Float = 128.0
//    let input_std: Float = 128.0
    
    let input_layer_name = "input"
    let output_layer_name = "softmax1"
//    let output_layer_name = "final_result"
    
    private static var AVCaptureStillImageIsCapturingStillImageContext = 0
    
    private var tf: TensorflowWrapper?

    private let session = AVCaptureSession()
    private let stillImageOutput = AVCaptureStillImageOutput()
    
    private let videoDataOutput = AVCaptureVideoDataOutput()
    
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    private var flashView: UIView?
    
    private var synth: AVSpeechSynthesizer?
    
    private var labelLayers: [CATextLayer] = []
    public var oldPredictionValues: [String: Float] = [:]
    
    public var labels: [String]?
    
    var isUsingFrontFacingCamera = false
    
    @IBOutlet weak var previewView: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        tf = TensorflowWrapper()
        tf?.delegate = self
        
        synth = AVSpeechSynthesizer()
        
        let load_status: Bool!
        if model_uses_memory_mapping {
            load_status = tf?.loadMemoryMappedModel(CameraViewController.model_file_name, modelFileType: CameraViewController.model_file_type)
        } else {
            load_status = tf?.loadModel(CameraViewController.model_file_name, modelFileType: CameraViewController.model_file_type)
        }
        
        if !load_status {
            fatalError("Couldn't load model")
        }
        
        labels = tf?.loadLabels(CameraViewController.labels_file_name, fileType: CameraViewController.labels_file_type) as? [String]
        if labels == nil || labels!.count == 0 {
            fatalError("Couldn't load labels")
        }
        
//        setupAVCapture()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        setupAVCapture()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
    }
    
    deinit {
        teardownAVCapture()
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        if context == &CameraViewController.AVCaptureStillImageIsCapturingStillImageContext {
            if let isCapturingStillImage = change?[.newKey] as? Bool, isCapturingStillImage {
                // do flash bulb like animation
                flashView = UIView(frame: previewView.frame)
                if let flashView = flashView {
                    flashView.backgroundColor = UIColor.white
                    flashView.alpha = 0.0
                    self.view.window?.addSubview(flashView)
                    
                    UIView.animate(withDuration: 0.4) {
                        flashView.alpha = 1.0
                    }
                }
            } else {
                UIView.animate(withDuration: 0.4, animations: {
                    self.flashView?.alpha = 0.0
                }) { finished in
                    self.flashView?.removeFromSuperview()
                    self.flashView = nil
                }
            }
        }
    }
    
    // MARK: - Actions
    @IBAction func dismiss(_ sender: UIButton) {
        dismiss(animated: true, completion: nil)
    }
    
    @IBAction func takePicture(_ sender: UIButton) {
        if session.isRunning {
            session.stopRunning()
            sender.setTitle("Continue", for: .normal)
            
            flashView = UIView(frame: previewView.frame)
            flashView?.backgroundColor = UIColor.white
            flashView?.alpha = 0.0
            self.view.window?.addSubview(flashView!)
            
            UIView.animate(withDuration: 0.2, animations: {
                self.flashView?.alpha = 1.0
            }, completion: { finished in
                UIView.animate(withDuration: 0.2, animations: {
                    self.flashView?.alpha = 0.0
                }, completion: { finished in
                    self.flashView?.removeFromSuperview()
                    self.flashView = nil
                })
            })
        } else {
            session.startRunning()
            sender.setTitle("Freeze Frame", for: .normal)
        }
    }
    
    // MARK: - Video Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        //        CFRetain(pixelBuffer)
        tf?.runCNN(onFrame: pixelBuffer, wantedInputChannels: wanted_input_channels, wantedInputWidth: wanted_input_width, wantedInputHeight: wanted_input_height, inputMean: input_mean, inputStd: input_std, inputLayerName: input_layer_name, outputLayerName: output_layer_name)
        //        CFRelease(pixelBuffer);
    }
    
    // MARK: Tensorflow Wrapper delegate
    func tensorflowWrapper(_ tensorflowWrapper: Any!, setPredictionValues newValues: [AnyHashable : Any]!) {
        let decayValue: Float = 0.75
        let updateValue: Float = 0.25
        let minimumThreshold: Float = 0.01
        
        var decayedPredictionValues: [String: Float] = [:]
        
        for (label, oldValue) in oldPredictionValues {
            let decayedPredictionValue = oldValue * decayValue
            if decayedPredictionValue > minimumThreshold {
                decayedPredictionValues[label] = decayedPredictionValue
            }
        }
        oldPredictionValues = decayedPredictionValues
        
        let newValues = newValues as! [String: NSNumber]
        
        for (label, newPredictionValueObject) in newValues {
            var oldPredictionValue = oldPredictionValues[label]
            if oldPredictionValue == nil {
                oldPredictionValue = 0.0
            }
            
            let newPredictionValue = newPredictionValueObject.floatValue
            let updatedPredictionValue = (oldPredictionValue! + (newPredictionValue * updateValue))
            
            oldPredictionValues[label] = updatedPredictionValue
        }
        
        var candidateLabels: [[String: Any]] = []
        for (label, oldPredictionValue) in oldPredictionValues {
            if oldPredictionValue > 0.05 {
                candidateLabels.append(["label": label, "value": oldPredictionValue])
            }
        }
        
        let sortedLabels = candidateLabels.sorted { (entry1, entry2) -> Bool in
            return (entry1["value"] as! Float) > (entry2["value"] as! Float)
        }
        
        let leftMargin: Float = 10.0
        let topMargin: Float = 10.0
        
        let valueWidth: Float = 48.0
        let valueHeight: Float = 26.0
        
        let labelWidth: Float = 246.0
        let labelHeight: Float = 26.0
        
        let labelMarginX: Float = 5.0
        let labelMarginY: Float = 5.0
        
        removeAllLabelLayers()
        
        var labelCount = 0
        for entry in sortedLabels {
            let label = entry["label"] as! String
            let value = entry["value"] as! Float
            
            let originY = (topMargin + ((labelHeight + labelMarginY) * Float(labelCount)))
            
            let valuePercentage = Int(roundf(value * 100.0))
            
            let valueOriginX = leftMargin
            let valueText = "\(valuePercentage)%"
            
            addLabelLayerWith(text: valueText, originX: valueOriginX, originY: originY, width: valueWidth, height: valueHeight, alignment: kCAAlignmentRight)
            
            let labelOriginX: Float = (leftMargin + valueWidth + labelMarginX)
            
            addLabelLayerWith(text: label.capitalized, originX: labelOriginX, originY: originY, width: labelWidth, height: labelHeight, alignment: kCAAlignmentLeft)
            
            if labelCount == 0 && value > 0.5 {
                speak(words: label.capitalized)
            }
            
            labelCount += 1
            if labelCount > 4 {
                break
            }
        }
    }
    
    private func removeAllLabelLayers() {
        for layer in labelLayers {
            layer.removeFromSuperlayer()
        }
        labelLayers.removeAll()
    }
    
    private func addLabelLayerWith(text: String, originX: Float, originY: Float, width: Float, height: Float, alignment: String) {
        let font: CFTypeRef = "Menlo-Regular" as CFTypeRef
        let fontSize: Float = 20.0
        
        let marginSizeX: Float = 5.0
        let marginSizeY: Float = 2.0
        
        let backgroundBounds = CGRect(x: CGFloat(originX), y: CGFloat(originY), width: CGFloat(width), height: CGFloat(height))

        let textBounds = CGRect(x: CGFloat(originX + marginSizeX),
                                y: CGFloat(originY + marginSizeY),
                                width: CGFloat(width - (marginSizeX * 2)),
                                height: CGFloat(height - (marginSizeY * 2)))
        
        let background = CATextLayer()
        background.backgroundColor = UIColor.black.cgColor
        background.opacity = 0.5
        background.frame = backgroundBounds
        background.cornerRadius = 5.0
        
        view.layer.addSublayer(background)
        labelLayers.append(background)
        
        let layer = CATextLayer()
        layer.foregroundColor = UIColor.white.cgColor
        layer.frame = textBounds
        layer.alignmentMode = alignment
        layer.isWrapped = true
        layer.font = font
        layer.fontSize = CGFloat(fontSize)
        layer.contentsScale = UIScreen.main.scale
        layer.string = text
        
        view.layer.addSublayer(layer)
        labelLayers.append(layer)
    }
    
    // MARK: - Voice
    private func speak(words: String) {
        if synth?.isSpeaking ?? false {
            return
        }
        
        let utterance = AVSpeechUtterance(string: words)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.75 * AVSpeechUtteranceDefaultSpeechRate
        synth?.speak(utterance)
    }
    
    // MARK: - Private
    private func setupAVCapture() {

        if UIDevice.current.userInterfaceIdiom == .phone {
            session.sessionPreset = .vga640x480
        } else {
            session.sessionPreset = .photo
        }
        
        do {
            let device = AVCaptureDevice.default(for: .video)
            let deviceInput = try AVCaptureDeviceInput(device: device!)
            
            isUsingFrontFacingCamera = false
            
            if session.canAddInput(deviceInput) {
                session.addInput(deviceInput)
            }
            
            stillImageOutput.addObserver(self, forKeyPath: "capturingStillImage", options: .new, context: &CameraViewController.AVCaptureStillImageIsCapturingStillImageContext)
            
            if session.canAddOutput(stillImageOutput) {
                session.addOutput(stillImageOutput)
            }
            
            videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as String) : NSNumber(value: kCMPixelFormat_32BGRA as UInt32)]
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            
            let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            
            if session.canAddOutput(videoDataOutput) {
                session.addOutput(videoDataOutput)
            }
            
            videoDataOutput.connection(with: .video)?.isEnabled = true
            
            previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer?.backgroundColor = UIColor.black.cgColor
            previewLayer?.videoGravity = .resizeAspectFill

            let rootLayer = previewView.layer
            rootLayer.masksToBounds = true
            previewLayer?.frame = rootLayer.bounds
//            previewLayer?.frame.size = previewView.frame.size
            rootLayer.addSublayer(previewLayer!)
            
            session.startRunning()
            
        } catch let nserror as NSError {

            let alert = UIAlertController(title: "Failed with error \(nserror.code)", message: nserror.localizedDescription, preferredStyle: .alert)
            
            let dismiss = UIAlertAction(title: "Dismiss", style: .default, handler: nil)
            alert.addAction(dismiss)
            
            present(alert, animated: true, completion: nil)
            
            teardownAVCapture()
        }
    }

    private func teardownAVCapture() {
//        stillImageOutput.removeObserver(self, forKeyPath: "isCapturingStillImage")
        stillImageOutput.removeObserver(self, forKeyPath: "capturingStillImage")
        previewLayer?.removeFromSuperlayer()
    }

    
}
