#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

// Define constants for image dimensions and detection thresholds
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.25;
const float CONFIDENCE_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.25;

// Define colors yellow, green, cyan, blue
const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };

// Detection structure (each detection has a class_id, confidence, box)
struct Detection {

	int class_id;
	float confidence;
	Rect box;
};

// Load classes in from 'classes.txt'
vector<string> load_classes() {

	vector<string> classes;
	ifstream file("classes/classes.txt");
	string line;

	while (getline(file, line)) {
		classes.push_back(line);
	}

	return classes;
}

// Load model from .onnx file 
void load_net(Net& net) {

	auto result = readNet("models/yolov5l_custom.onnx");

	// Use CUDA GPU 
	result .setPreferableBackend(DNN_BACKEND_CUDA);
	result.setPreferableTarget(DNN_TARGET_CUDA);

	if (cuda::getCudaEnabledDeviceCount() > 0) {
		cout << "CUDA devices found, using GPU" << endl;

		// Check GPU used
		int deviceId = cuda::getDevice();
		cuda::DeviceInfo deviceInfo(deviceId);
		cout << "Using device " << deviceInfo.name() << endl;
	}
	else {
		cout << "No CUDA devices found, using CPU" << endl;
	}

	net = result;
}

Mat preprocess(const Mat& source) {

	// Get frame dimensions and find max
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);

	// Creates a new 'result' image/frame with zeroes (black pixels),
	// of dimensions 'MAX x MAX', with 3 channels (color image)
	Mat result = Mat::zeros(_max, _max, CV_8UC3);

	// Copy source image to the top left corner of the result image
	source.copyTo(result(Rect(0, 0, col, row)));

	return result;
}

void detect(Mat& image, Net& net, vector<Detection>& output, const vector<string>& className) {

	Mat blob;
	vector<Mat> outputs;

	auto input_frame = preprocess(image);

	blobFromImage(input_frame, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	// Model dimensions - use Netron.app to find
	const int dimensions = 7;
	const int rows = 25200;

	float x_factor = input_frame.cols / INPUT_WIDTH;
	float y_factor = input_frame.rows / INPUT_HEIGHT;
	float* data = (float*)outputs[0].data;

	vector<int> class_ids;
	vector<float> confidences;
	vector<Rect> boxes;

	// Iterate through detections
	for (int i = 0; i < rows; i++) {

		// Confidence value is at the 5th position of the detection array
		float confidence = data[4];

		if (confidence >= CONFIDENCE_THRESHOLD) {

			// Points to data[5], where the first class score is located in the detection array 
			float* classes_scores = data + 5;

			// Creates a 'scores' matrix of height 1 and width the size of the number of classes
			// classes_scores points to the class scores in the detection array
			Mat scores(1, className.size(), CV_32FC1, classes_scores);
			Point class_id;
			double max_class_score;

			// minMaxLoc(scores matrix, minimum score, store max score here, min loc, store max loc here)
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

			if (max_class_score > SCORE_THRESHOLD) {

				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);

				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];

				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);

				boxes.push_back(Rect(left, top, width, height));
			}
		}

		data += 7;
	}

	// Non-Maximum Supression
	vector<int> nms_result;
	
	NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

	for (int i = 0; i < nms_result.size(); i++) {

		int idx = nms_result[i];
		Detection result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}

}

int main() {

	vector<string> classes = load_classes();


	// Video path
	Mat frame;
	VideoCapture capture("videos/test_video1.mp4");

	if (!capture.isOpened()) {
		
		cout << "Error opening video file." << endl;
		return -1;
	}

	Net net;
	load_net(net);

	float fps = -1;

	while (true) {

		capture.read(frame);

		if (frame.empty()) {
			cout << "End of video." << endl;
			break;
		}

		vector<Detection> output;
		detect(frame, net, output, classes);

		int detections = output.size();

		for (int i = 0; i < detections; i++) {

			auto detection = output[i];
			auto box = detection.box;
			auto classId = detection.class_id;
			float confidenceLvl = detection.confidence;
			const auto color = colors[classId % colors.size()];

			rectangle(frame, box, color, 3);
			rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);

			// Show class name and confidence score
			stringstream ss;
			ss << fixed << setprecision(2) << confidenceLvl;
			string label = classes[classId] + " " + ss.str();

			putText(frame, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		}

		// Implement FPS counter here

		imshow("Video", frame);

		if (waitKey(1) != -1) {

			capture.release();
			cout << "Terminated by user." << endl;
			break;
		}
	}

	return 0;
}