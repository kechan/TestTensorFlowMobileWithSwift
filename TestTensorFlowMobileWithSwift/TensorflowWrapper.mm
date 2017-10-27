//
//  TensorflowWrapper.m
//  TestTensorFlowMobileWithSwift
//
//  Created by Kelvin C on 10/20/17.
//  Copyright © 2017 Kelvin Chan. All rights reserved.
//

#import "TensorflowWrapper.h"
#import "ios_image_load.h"
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include "tensorflow_utils.h"

namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return (int)ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}  // namespace


using namespace std;

@implementation TensorflowWrapper {
    std::vector<std::string> labels;
    std::unique_ptr<tensorflow::Session> tf_session;
    std::unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;
}

@synthesize delegate;

- (NSString*) RunInferenceOnImage:(NSString*)filename {
    tensorflow::SessionOptions options;
    
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        return [NSString stringWithFormat: @"Session create failed - %s",
                status_string.c_str()];
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "Session created.";
    
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "Graph created.";
    
// NSString* network_path = FilePathForResourceName(@"tensorflow_inception_graph", @"pb");
    NSString* network_path = FilePathForResourceName(@"frozen_inception_v3", @"pb");
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    LOG(INFO) << "Creating session.";
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
        return @"";
    }
    
    // Read the label list
//    NSString* labels_path = FilePathForResourceName(@"imagenet_comp_graph_label_strings", @"txt");
    NSString* labels_path = FilePathForResourceName(@"imagenet_slim_labels", @"txt");
    std::vector<std::string> label_strings;
    std::ifstream t;
    t.open([labels_path UTF8String]);
    std::string line;
    while(t){
        std::getline(t, line);
        label_strings.push_back(line);
    }
    t.close();
    
    // Read the Grace Hopper image.
    NSString* image_path = FilePathForResourceName(filename, @"jpg");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
                                                                  [image_path UTF8String], &image_width, &image_height, &image_channels);
//    const int wanted_width = 224;
//    const int wanted_height = 224;
    const int wanted_width = 299;
    const int wanted_height = 299;
    const int wanted_channels = 3;
//    const float input_mean = 117.0f;
//    const float input_std = 1.0f;
    const float input_mean = 0.0f;
    const float input_std = 255.0f;

    assert(image_channels >= wanted_channels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({
        1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8* in = image_data.data();
    // tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    NSString* result = [network_path stringByAppendingString: @" - loaded!"];
    result = [NSString stringWithFormat: @"%@ - %lu, %s - %dx%d", result,
              label_strings.size(), label_strings[0].c_str(), image_width, image_height];
    
    std::string input_layer = "input";
//    std::string output_layer = "output";
    std::string output_layer = "InceptionV3/Predictions/Reshape_1";
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
                                                 {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        tensorflow::LogAllRegisteredKernels();
        result = @"Error running model";
        return result;
    }
    tensorflow::string status_string = run_status.ToString();
    result = [NSString stringWithFormat: @"%@ - %s", result,
              status_string.c_str()];
    
    tensorflow::Tensor* output = &outputs[0];
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    std::vector<std::pair<float, int> > top_results;
    GetTopN(output->flat<float>(), kNumResults, kThreshold, &top_results);
    
    std::stringstream ss;
    ss.precision(3);
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        
        ss << index << " " << confidence << "  ";
        
        // Write out the result as a string
        if (index < label_strings.size()) {
            // just for safety: theoretically, the output is under 1000 unless there
            // is some numerical issues leading to a wrong prediction.
            ss << label_strings[index];
        } else {
            ss << "Prediction: " << index;
        }
        
        ss << "\n";
    }
    
    LOG(INFO) << "Predictions: " << ss.str();
    
    tensorflow::string predictions = ss.str();
    result = [NSString stringWithFormat: @"%@ - %s", result,
              predictions.c_str()];
    
    return result;
    
}

//NSString* FilePathForResourceName(NSString* name, NSString* extension) {
//    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
//    if (file_path == NULL) {
//        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
//        << [extension UTF8String] << "' in bundle.";
//    }
//    return file_path;
//}

//bool PortableReadFileToProto(const std::string& file_name,
//                             ::google::protobuf::MessageLite* proto) {
//    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
//                                                             new IfstreamInputStream(file_name));
//    stream.SetOwnsCopyingStream(true);
//    // TODO(jiayq): the following coded stream is for debugging purposes to allow
//    // one to parse arbitrarily large messages for MessageLite. One most likely
//    // doesn't want to put protobufs larger than 64MB on Android, so we should
//    // eventually remove this and quit loud when a large protobuf is passed in.
//    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
//    // Total bytes hard limit / warning limit are set to 1GB and 512MB
//    // respectively.
//    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
//    return proto->ParseFromCodedStream(&coded_stream);
//}

// sorted by confidence in descending order.
//static void GetTopN(
//                    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
//                    Eigen::Aligned>& prediction,
//                    const int num_results, const float threshold,
//                    std::vector<std::pair<float, int> >* top_results) {
//    // Will contain top N results in ascending order.
//    std::priority_queue<std::pair<float, int>,
//    std::vector<std::pair<float, int> >,
//    std::greater<std::pair<float, int> > > top_result_pq;
//    
//    const long count = prediction.size();
//    for (int i = 0; i < count; ++i) {
//        const float value = prediction(i);
//        
//        // Only add it if it beats the threshold and has a chance at being in
//        // the top N.
//        if (value < threshold) {
//            continue;
//        }
//        
//        top_result_pq.push(std::pair<float, int>(value, i));
//        
//        // If at capacity, kick the smallest value out.
//        if (top_result_pq.size() > num_results) {
//            top_result_pq.pop();
//        }
//    }
//    
//    // Copy to output vector and reverse into descending order.
//    while (!top_result_pq.empty()) {
//        top_results->push_back(top_result_pq.top());
//        top_result_pq.pop();
//    }
//    std::reverse(top_results->begin(), top_results->end());
//}

-(void)runCNNOnFrame:(CVPixelBufferRef)pixelBuffer wantedInputChannels:(int)wantedInputChannels
    wantedInputWidth:(int)wantedInputWidth
   wantedInputHeight:(int)wantedInputHeight
           inputMean:(float)inputMean
            inputStd:(float)inputStd
      inputLayerName:(NSString*)inputLayerName
     outputLayerName:(NSString*)outputLayerName{
    
    std::string input_layer_name = std::string([inputLayerName UTF8String]);
    std::string output_layer_name = std::string([outputLayerName UTF8String]);
    
    assert(pixelBuffer != NULL);
    
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    int doReverseChannels;
    if (kCVPixelFormatType_32ARGB == sourcePixelFormat) {
        doReverseChannels = 1;
    } else if (kCVPixelFormatType_32BGRA == sourcePixelFormat) {
        doReverseChannels = 0;
    } else {
        assert(false);  // Unknown source format
    }
    
    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char *sourceBaseAddr =
    (unsigned char *)(CVPixelBufferGetBaseAddress(pixelBuffer));
    int image_height;
    unsigned char *sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    const int image_channels = 4;
    
    assert(image_channels >= wantedInputChannels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape(
                                                            {1, wantedInputHeight, wantedInputWidth, wantedInputChannels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8 *in = sourceStartAddr;
    float *out = image_tensor_mapped.data();
    for (int y = 0; y < wantedInputHeight; ++y) {
        float *out_row = out + (y * wantedInputWidth * wantedInputChannels);
        for (int x = 0; x < wantedInputWidth; ++x) {
            const int in_x = (y * image_width) / wantedInputWidth;
            const int in_y = (x * image_height) / wantedInputHeight;
            tensorflow::uint8 *in_pixel =
            in + (in_y * image_width * image_channels) + (in_x * image_channels);
            float *out_pixel = out_row + (x * wantedInputChannels);
            for (int c = 0; c < wantedInputChannels; ++c) {
                out_pixel[c] = (in_pixel[c] - inputMean) / inputStd;
            }
        }
    }
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    
    if (tf_session.get()) {
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = tf_session->Run(
                                                        {{input_layer_name, image_tensor}}, {output_layer_name}, {}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed:" << run_status;
        } else {
            tensorflow::Tensor *output = &outputs[0];
            auto predictions = output->flat<float>();
            
            NSMutableDictionary *newValues = [NSMutableDictionary dictionary];   // [String: NSNumber]
            for (int index = 0; index < predictions.size(); index += 1) {
                const float predictionValue = predictions(index);
                if (predictionValue > 0.05f) {
                    std::string label = labels[index % predictions.size()];
                    NSString *labelObject = [NSString stringWithUTF8String:label.c_str()];
                    NSNumber *valueObject = [NSNumber numberWithFloat:predictionValue];
                    [newValues setObject:valueObject forKey:labelObject];
                }
            }
            dispatch_async(dispatch_get_main_queue(), ^(void) {
                if (delegate != nil && [delegate respondsToSelector:@selector(tensorflowWrapper:setPredictionValues:)]) {
                    // [self setPredictionValues:newValues];  // TODO: how to pass prediction values back to viewcontroller?
                    [(id<TensorflowWrapperDelegate>)delegate tensorflowWrapper:self setPredictionValues:newValues];
                }
            });
        }
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

-(bool)loadMemoryMappedModel:(NSString*)modelFileName modelFileType:(NSString*)modelFileType {
    tensorflow::Status load_status = LoadMemoryMappedModel(modelFileName, modelFileType, &tf_session, &tf_memmapped_env);
    return load_status.ok();
}

-(bool)LoadModel:(NSString*)modelFileName modelFileType:(NSString*)modelFileType {
    tensorflow::Status load_status = LoadModel(modelFileName, modelFileType, &tf_session);
    return load_status.ok();
}

-(NSArray*)LoadLabels:(NSString*)fileName fileType:(NSString*)fileType {
//    std::vector<std::string> labels;
    tensorflow::Status labels_status = LoadLabels(fileName, fileType, &labels);
    
    NSMutableArray *label_strings = [NSMutableArray new];
    
    for (auto str: labels) {
        id nsstr = [NSString stringWithUTF8String:str.c_str()];
        [label_strings addObject:nsstr];
    }
    
    // For Wrapper use
//    tensorflow::Status labels_status = LoadLabels(fileName, fileType, &labels);
//    if (!labels_status.ok()) {
//        LOG(FATAL) << "Couldn't load labels: " << labels_status;
//    }
    
    return label_strings;
}

//tensorflow::Status LoadLabels(NSString* file_name, NSString* file_type,
//                              std::vector<std::string>* label_strings) {
//    // Read the label list
//    NSString* labels_path = FilePathForResourceName(file_name, file_type);
//    if (!labels_path) {
//        LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
//        << [file_type UTF8String];
//        return tensorflow::errors::NotFound([file_name UTF8String],
//                                            [file_type UTF8String]);
//    }
//    std::ifstream t;
//    t.open([labels_path UTF8String]);
//    std::string line;
//    while (t) {
//        std::getline(t, line);
//        label_strings->push_back(line);
//    }
//    t.close();
//    return tensorflow::Status::OK();
//}

@end
