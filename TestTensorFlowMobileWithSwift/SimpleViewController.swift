//
//  ViewController.swift
//  TestTensorFlowMobileWithSwift
//
//  Created by Kelvin C on 10/20/17.
//  Copyright © 2017 Kelvin Chan. All rights reserved.
//

import UIKit

class SimpleViewController: UIViewController {

    @IBOutlet weak var resultTextView: UITextView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    @IBAction func runModel(_ sender: UIButton) {
        let tf = TensorflowWrapper()
        let result = tf.runInference(onImage: "grace_hopper")  // Always Grace Hopper
        
        resultTextView.text = result 
    }
    
    @IBAction func dismiss(_ sender: UIButton) {
        dismiss(animated: true, completion: nil)
    }
}

