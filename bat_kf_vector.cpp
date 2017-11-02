/*
CS585 Image and Video Computing Fall 2017
Assignment 04
This code tracks objects in binary images
Authors: G Sivaperumal, Shruti Kannan, Prateek Mehta
*/

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stack>
#include <filesystem>
#include <cmath>


namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

void file_processing(string filename_det, vector <vector<int>>& data_store);
void init_kalman(KalmanFilter &KF);
void get_next_center(Point &estimated_bat, vector <int> &predicted_bat, vector <vector <int>> &file_data1, int &index);
void convertFileToMat(String filename, Mat& labelled, Mat& binary);
void color_bat(vector<vector<int>>current_frame, vector<vector<int>>next_frame, vector <vector <double>> &colors, Mat& output, vector <KalmanFilter> &kf_vector);
void operations(Mat &output);

int main() 
{
	//VideoWriter to create output video
	VideoWriter video("bat_track.avi", CV_FOURCC('M', 'J', 'P', 'G'), 3, Size(1024, 1024));

	// Storing file names of Segmentation and Localization in vectors
	vector<string> filename_loc;
	vector<string> filename_seg;
	std::string path = "Localization_Bats/";
	std::string path1 = "Seg_images/"; // Binary images folder
	const fs::directory_iterator end{}; 
	int no_frames = 0;
	for (fs::directory_iterator iter{ path }; iter != end; ++iter) {
		filename_loc.push_back(iter->path().string()); // Contains Localization filenames
		no_frames++;
	}
	for (fs::directory_iterator iter{ path1 }; iter != end; ++iter) {
		filename_seg.push_back(iter->path().string()); // Contains Segmentation filenames
	}

	// Create vector of KF objects and Initialize Kalman parameters
	vector <KalmanFilter> kf_vector;
	for (int i = 0; i < no_frames; i++) {
		KalmanFilter KF(2, 2, 0);
		init_kalman(KF);
		kf_vector.push_back(KF); // State dim = 2, Measurement dim = 2
	}

	vector <vector <int>> file_data, file_data1, file_data2; // Vectors to store centers of current and next frame

	// Declarations
	int tmp = 0, tmp1 = 0;
	Mat_<float> measurement(2, 1);
	vector <int> current_bat, predicted_bat;
	Point estimated_bat(0.0, 0.0), predictPt(0.0, 0.0);
	Mat estimation, prediction, labelled0, binary0, labelled, binary;
	Mat1b src0, src;
	Mat output0, output;
	std::vector<cv::Point> blob0, blob;
	int index = 0;
	vector <vector <double>> colors;
	vector <double> c;

	file_processing(filename_loc[0], file_data); // Initialising frame#0 Localization file to vector of centers
	
	c.clear(); // Vector for colours
	for (int i = 0; i < file_data.size(); i++)
	{
		c.push_back(255 * (rand() / (1.0 + RAND_MAX))); // Creating random colors and storing in vector
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		// cout << "intial_color_allocation:" << c.at(0) << " " << c.at(1) << " " << c.at(2) << " " << endl;
		colors.push_back(c);
		c.clear();
	}
	
	for (int current_loc = 0; current_loc < no_frames - 1; current_loc++)  // Loop over Localization files
	{
		// Convert Localization file (with centers) to vector
		file_data1.clear();
		file_processing(filename_loc[current_loc + 1], file_data1); // Next frame Localization file to vector of centers

		prediction = kf_vector[current_loc].predict();
		predictPt.x = prediction.at<float>(0);
		predictPt.y = prediction.at<float>(1);

		binary0 = imread(filename_seg[current_loc], CV_LOAD_IMAGE_GRAYSCALE); // Reading the binary images
		binary = imread(filename_seg[current_loc + 1], CV_LOAD_IMAGE_GRAYSCALE); 
		cvtColor(binary0, output0, CV_GRAY2BGR); // Copying the binary to output images
		cvtColor(binary, output, CV_GRAY2BGR);

		//operations(output); // if video is reversed (1/3)
		//operations(output0); // if video is reversed (2/3)

		for (int i = 0; i < file_data.size(); i++) // Loop over bats in each Localization file
		{ 
			tmp1++;
			measurement(0) = file_data[i][0];
			measurement(1) = file_data[i][1];
			current_bat.push_back(file_data[i][0]);
			current_bat.push_back(file_data[i][1]);
			// cout << "measurement: " << measurement(0) << "," << measurement(1) << endl;

			estimation = kf_vector[current_loc].correct(measurement); // Get estimate for the bat in next frame
			estimated_bat.x = estimation.at<float>(0); // Estimation of KF as a Point
			estimated_bat.y = estimation.at<float>(1);
			// cout << "estimation: " << estimated_bat.x << "," << estimated_bat.y << endl;

			get_next_center(estimated_bat, predicted_bat, file_data1, index); // Likely next bat center
			// cout << "MF predicted Bat: " << predicted_bat[0] << "," << predicted_bat[1] << endl;

			file_data1.erase(file_data1.begin() + index); // Remove bat from this vector to check if extra bats exist
			file_data2.push_back(predicted_bat);
			current_bat.clear();
			if (file_data1.size() == 0) { // If next frame has lesser bats than previous frame then break loop
				cout << "     Next frame has lesser bats!" << endl;
				break;
			}
		}

		file_data2.insert(std::end(file_data2), std::begin(file_data1), std::end(file_data1)); // Insert extra bats to this vector
		color_bat(file_data, file_data2,colors, output, kf_vector); // Colour the bats
		cv::namedWindow("output current frame", WINDOW_NORMAL);
		cv::imshow("output current frame", output0);
		cv::namedWindow("output next frame", WINDOW_NORMAL);
		cv::imshow("output next frame", output);
		file_data = file_data2;
		file_data2.clear();
		
		//write the output
//		if (current_loc == 0)
//			video.write(output0);
		video.write(output);

		//char key = waitKey(0);
	}
	video.release();
	return 0;
}

void operations(Mat &output) {  // In case frames do not appear properly
	transpose(output, output);
	flip(output, output, 1);
	transpose(output, output);
	flip(output, output, 1);
	transpose(output, output);
	flip(output, output, 0);
	flip(output, output, 1);
	return;
}

void color_bat(vector<vector<int>>current_frame, vector<vector<int>>next_frame, vector <vector <double>> &colors, Mat& output, vector <KalmanFilter> &kf_vector)
{
	vector<double> color;
	int bat_no_diff = next_frame.size() - current_frame.size();
	cout << "difference in bats:" << bat_no_diff << endl;
	vector<double> single_pixel;

	cout << "colors_size" << colors.size() << endl;
	for (int i = 0; i < current_frame.size(); i++)
	{  
		if (i >= next_frame.size()) {
			continue;
		}
		else
		{
			single_pixel = colors.at(i);
			cout << "i: " << i << endl;
			line(output, Point(current_frame[i][0], current_frame[i][1]), Point(next_frame[i][0], next_frame[i][1]),
				Scalar(single_pixel.at(0), single_pixel.at(1), single_pixel.at(2)), 2, LINE_8);
			single_pixel.clear();
		}
	}
	if (bat_no_diff > 0)
		for (int i = 0; i < bat_no_diff;i++)
		{
			cout << "adding color" << endl;
			color.push_back(255 * (rand() / 1.0 + RAND_MAX));
			color.push_back(255 * (rand() / 1.0 + RAND_MAX));
			color.push_back(255 * (rand() / 1.0 + RAND_MAX));

			colors.push_back(color);
			color.clear();
			KalmanFilter KF(2, 2, 0);
			init_kalman(KF);
			kf_vector.push_back(KF);
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

void get_next_center(Point &estimated_bat, vector <int> &predicted_bat, vector <vector <int>> &file_data1, int &index) {
	float difference, distance = 0;
	for (int i = 0; i < file_data1.size(); i++) { // Loop through centers of next file
		Point nfp(file_data1[i][0], file_data1[i][1]);
		if (i == 0) {
			difference = sqrt(pow((estimated_bat.x - nfp.x), 2) + pow((estimated_bat.y - nfp.y), 2)); // compute distance between bats
			index = 0;			
		}
		else {
			distance = sqrt(pow((estimated_bat.x - nfp.x), 2) + pow((estimated_bat.y - nfp.y), 2));
			if (distance < difference) { // check for shortest distance
				difference = distance; // store value, index
				index = i;
			}
		}
	}
	predicted_bat = file_data1[index];
}

void init_kalman(KalmanFilter &KF) {
	setIdentity(KF.transitionMatrix); // A
	setIdentity(KF.measurementMatrix); // H
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5)); // Q
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1)); // R
	setIdentity(KF.errorCovPost, Scalar::all(1)); // P
	KF.statePre.at<float>(0) = 0; // Init parameters
	KF.statePre.at<float>(1) = 0;
}

void file_processing(string filename_det, vector <vector<int>>& data_store) {
	ifstream infile(filename_det);
	if (!infile) {
		cout << "Error reading file!";
		return;
	}
	data_store.clear();
	//read the comma separated values into a vector of vector of ints
	while (infile) {
		string s;
		if (!getline(infile, s)) break;
		istringstream ss(s);
		vector <int> datarow;
		while (ss) {
			string srow;
			//cout << "1aa" << endl;
			int sint;
			if (!getline(ss, srow, ',')) break;
			sint = atoi(srow.c_str()); //convert string to int
			datarow.push_back(sint);
		}
		data_store.push_back(datarow);
	}
}
