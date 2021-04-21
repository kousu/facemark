#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/face/facemark_train.hpp"

const float confidenceThreshold = 0.80;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

using namespace cv;
using namespace face;

struct Conf {
    cv::String model_path;
    double scaleFactor;
    Conf(cv::String s, double d){
        model_path = s;
        scaleFactor = d;
        face_detector.load(model_path);
    };
    CascadeClassifier face_detector;
};

cv::Rect2f expandRect(cv::Rect2f rect, cv::Size frameSize, uint pixelsToExpandx, uint pixelsToExpandy) {

    //expand rectangle
    rect.x -= pixelsToExpandx;
    rect.y -= pixelsToExpandy;
    rect.width += pixelsToExpandx * 2;
    rect.height += pixelsToExpandy * 2;


    //make sure it fits the frame
    rect.x = (rect.x > 0) ? rect.x : 0;
    rect.y = (rect.y > 0) ? rect.y : 0;
    rect.height = ((rect.height + rect.y) > frameSize.height) ? (frameSize.height - rect.y) : rect.height;
    rect.width = ((rect.width + rect.x) > frameSize.width) ? (frameSize.width - rect.x) : rect.width;

    return rect;
}


bool myDetector(InputArray image, OutputArray faces, cv::dnn::Net *net){

  cv::Mat frameResized;
  const cv::Size frameSize = image.getMat().size();

  

  cv::resize(image.getMat(), frameResized, cv::Size(300, 300));

  if(image.channels()!=3) {
      // convert greyscale to RGB?
      cvtColor(frameResized,frameResized,COLOR_GRAY2BGR);
  }

  //convert to blob
  cv::Mat inputBlob = cv::dnn::blobFromImage(frameResized, 1, cv::Size(300, 300), meanVal, false, false);

  // Detect faces
  net->setInput(inputBlob, "data");
  cv::Mat detection = net->forward("detection_out");

  //reshape output
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

  //arrays
  std::vector<cv::Rect> _faces;

  //look through first 10 results (to save time)
  for (int i = 0; i < detectionMat.rows; i++)
  {
      float confidence = detectionMat.at<float>(i, 2);
      //Make sure confidence is above threshold
      if (confidence >= confidenceThreshold)
      {
          //get face rect
          int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameSize.width);
          int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameSize.height);
          int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameSize.width);
          int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameSize.height);

          cv::Rect faceRect(cv::Point(x1, y1), cv::Point(x2, y2));

          //MAKE BOX SQUARE
          auto x = faceRect.size();
          auto y = x.height - x.width;
          faceRect = expandRect(faceRect, frameSize, y/2, 0);

          //add to the arrays
          _faces.push_back(faceRect);
      }
  }

    Mat(_faces).copyTo(faces);
    return true;
}


int show(cv::dnn::Net* net,String model,String image) {

    // Create an instance of Facemark
    cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel(model);

    cv::Mat frame1;
    frame1 = imread(image.c_str());
    if (frame1.empty()) return 1; // end of video stream

    std::vector<cv::Rect> faces;
    myDetector(frame1, faces, net);

    std::vector< std::vector<cv::Point2f> > landmarks;

    // Run landmark detector
    bool success = facemark->fit(frame1, faces, landmarks);
    if (success) {
        for (int j = 0; j < faces.size(); j++) {
            face::drawFacemarks(frame1, landmarks[j], Scalar(0, 0, 255));
            cv::rectangle(frame1, faces[j], cv::Scalar(0, 255, 0), 2, 4);

        }

    }
    imshow("result", frame1);
    waitKey(0);
    
    return 0;
}

int track(cv::dnn::Net* net,String model) {

    // Create an instance of Facemark
    cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel(model);

    cv::VideoCapture cap;

    if (!cap.open(0))
        return 0;

    cv::Mat frame1, frame2;
    while (true)
    {
        cap >> frame1;
        if (frame1.empty()) break; // end of video stream

        std::vector<cv::Rect> faces;
        myDetector(frame1, faces, net);


        std::vector< std::vector<cv::Point2f> > landmarks;


        // Run landmark detector
        bool success = facemark->fit(frame1, faces, landmarks);
        if (success) {
            for (int j = 0; j < faces.size(); j++) {
                face::drawFacemarks(frame1, landmarks[j], Scalar(0, 0, 255));
                cv::rectangle(frame1, faces[j], cv::Scalar(0, 255, 0), 2, 4);

            }

        }
        imshow("result", frame1);
        waitKey(1);
        
    }
    
    return 0;
}

void printHelp() {
    std::cout << "FacemarkTrain" << " train <dnn location> <dataset folder> <output model file> " << std::endl;
    std::cout << "FacemarkTrain" << " test  <dnn location> <model> <image> " << std::endl;
    std::cout << "FacemarkTrain" << " track <dnn location> <model> " << std::endl << std::endl;

    std::cout << "Commands:" << std::endl;
    std::cout << "train - Train the Facemark algorithm to work with the dnn" << std::endl;
    std::cout << "test - Test the Facemark algorithm with a picture" << std::endl;
    std::cout << "track - Test the Facemark algorithm with a webcam" << std::endl << std::endl;

    std::cout << "Parameters:" << std::endl;
    std::cout << "dnn location - Location of the model.prototxt & model.caffemodel files" << std::endl;
    std::cout << "dataset folder - Location of the dataset with the annotations.txt & images.txt files." << std::endl;
    std::cout << "model - File to load into Facemark as input" << std::endl;
    std::cout << "output model file - File to save into Facemark as output" << std::endl;
    std::cout << "image - File to load to test Facemark on" << std::endl << std::endl;
}

int main(int argc, char* argv[]){
    try
    {
        if(argc < 2) {
            throw std::invalid_argument("Invalid usage");
        }
    
        auto command = String(argv[1]);
        auto caffeLocation = String(argv[2]);

        const std::string caffeConfigFile = caffeLocation + "model.prototxt";
        const std::string caffeWeightFile = caffeLocation + "model.caffemodel";
        cv::dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

        if(command == "train") {
            if(argc != 5) throw std::invalid_argument("Invalid usage");

            auto dataset = String(argv[3]);
            auto model = String(argv[4]);
        
            FacemarkLBF::Params params;
            params.stages_n=5; // amount of refinement stages
            params.tree_n=6; // number of tree in the model for each landmark point
            params.tree_depth=5; //he depth of decision tree
            params.cascade_face = "none"; //cascade file for default detector
            params.initShape_n = 10; //multiplier for augment the training data
            params.bagging_overlap = 0.4; //overlap ratio for training the LBF feature
            params.save_model = true;
            params.verbose = true;
            params.seed = 0; //seed

            params.model_filename = model;

            Ptr<FacemarkTrain> facemark = FacemarkLBF::create(params);
      

            std::vector<String> images_train;
            std::vector<String> landmarks_train;
            loadDatasetList(dataset+"/images.txt",dataset+"/annotations.txt",images_train,landmarks_train);

            Mat image;
            std::vector<Point2f> facial_points;

            facemark->setFaceDetector((FN_FaceDetector)myDetector,&net); // we must guarantee proper lifetime of "config" object

            std::cout << "Adding Data." << std::endl;
            for(size_t i=0;i<images_train.size();i++){
                image = imread(images_train[i].c_str());
                loadFacePoints(landmarks_train[i],facial_points);
                facemark->addTrainingSample(image, facial_points);
                std::cout << "Adding Data. "  << std::to_string(i+1) << "/" << images_train.size() << std::endl;
            }

            destroyAllWindows();
            std::cout << "Training..." << std::endl;
            facemark->training();
        }
        else if(command == "test") {
            if (argc != 5) throw std::invalid_argument("Invalid usage");
            std::cout << "Testing..." << std::endl;
            auto model = String(argv[3]);
            auto image = String(argv[4]);
            show(&net, model, image);
        }
        else if(command == "track") {
            if(argc != 4) throw std::invalid_argument("Invalid usage");
            auto model = String(argv[3]);
            track(&net, model);
            return 0;

        }
        else if (command == "help") {
            printHelp();
        }
        
    }
    catch (const std::invalid_argument& e) {
        std::cout << "error: " << e.what() << std::endl;
        printHelp();
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cout << "error: " << e.what() << std::endl;
        return 1;
    }
  
    
	return 0;
}
