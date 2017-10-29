//
//  TensorflowWrapper.h
//  TestTensorFlowMobileWithSwift
//
//  Created by Kelvin C on 10/20/17.
//  Copyright © 2017 Kelvin Chan. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>

@protocol TensorflowWrapperDelegate;

@interface TensorflowWrapper : NSObject

-(NSString*)RunInferenceOnImage:(NSString*)filename;

-(void)runCNNOnFrame:(CVPixelBufferRef)pixelBuffer wantedInputChannels:(int)wantedInputChannels
    wantedInputWidth:(int)wantedInputWidth
   wantedInputHeight:(int)wantedInputHeight
           inputMean:(float)inputMean
            inputStd:(float)inputStd
      inputLayerName:(NSString*)inputLayerName
     outputLayerName:(NSString*)outputLayerName;

-(bool)loadMemoryMappedModel:(NSString*)modelFileName modelFileType:(NSString*)modelFileType;
-(bool)LoadModel:(NSString*)modelFileName modelFileType:(NSString*)modelFileType;
-(NSArray*)LoadLabels:(NSString*)fileName fileType:(NSString*)fileType;


@property (nonatomic, weak) id <TensorflowWrapperDelegate> delegate;
@end

@protocol TensorflowWrapperDelegate <NSObject>
@optional
- (void)tensorflowWrapper:(id)tensorflowWrapper setPredictionValues:(NSDictionary*)newValues;
@end
