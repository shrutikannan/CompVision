#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

void convertFileToMat(String filename, Mat& labelled, Mat& binary);

int main() {

	vector<string> filename_seg;
	std::string path1 = "Segmentation_Bats/";
	const fs::directory_iterator end{}; //
	int no_frames = 0;
	for (fs::directory_iterator iter{ path1 }; iter != end; ++iter) { // Store filenames in vector
		filename_seg.push_back(iter->path().string());
		no_frames++; // Check no of images
	}
	cout << "#images: " << no_frames << endl;

	for (int i = 0; i < filename_seg.size(); i++) { // Loop through filenames
		cout << "Current file: " << i << endl;
		char file_name[100];
		Mat labelled, binary;
		sprintf(file_name, "./Seg_images/img%d.jpg", i + 1); // Path to store the images and filename
		convertFileToMat(filename_seg[i], labelled, binary); // Create binary for each file
		imwrite(file_name, binary); // Write binary to filename
	}
}

void convertFileToMat(String filename, Mat& labelled, Mat& binary) {
	ifstream infile(filename);
	vector <vector <int>> data;
	if (!infile) {
		cout << "Error reading file!";
		return;
	}
	//read the comma separated values into a vector of vector of ints
	while (infile) {
		string s;
		if (!getline(infile, s)) break;
		istringstream ss(s);
		vector <int> datarow;
		while (ss) {
			string srow;
			int sint;
			if (!getline(ss, srow, ',')) break;
			sint = atoi(srow.c_str()); //convert string to int
			datarow.push_back(sint);
		}
		data.push_back(datarow);
	}

	//construct the labelled matrix from the vector of vector of ints
	labelled = Mat::zeros(data.size(), data.at(0).size(), CV_8UC1);
	for (int i = 0; i < labelled.rows; ++i)
		for (int j = 0; j < labelled.cols; ++j)
			labelled.at<uchar>(i, j) = data.at(i).at(j);

	//construct the binary matrix from the labelled matrix
	binary = Mat::zeros(labelled.rows, labelled.cols, CV_8UC1);
	for (int i = 0; i < labelled.rows; ++i) {
		for (int j = 0; j < labelled.cols; ++j) {
			binary.at<uchar>(i, j) = (labelled.at<uchar>(i, j) == 0) ? 0 : 255;
		}
	}
}