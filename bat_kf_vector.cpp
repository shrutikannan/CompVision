///*
//CS585 Image and Video Computing Fall 2017
//Assignment 04
//Authors: G Sivaperumal, Shruti Kannan, Prateek Mehta

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
	// Finding no of files in Loc_files
	vector<string> filename_loc;
	vector<string> filename_seg;
	std::string path = "Localization_Bats/";
	std::string path1 = "Seg_images/"; //1
	const fs::directory_iterator end{}; //
	int no_frames = 0;
	for (fs::directory_iterator iter{ path }; iter != end; ++iter) {
		filename_loc.push_back(iter->path().string());
		no_frames++;
	}
	for (fs::directory_iterator iter{ path1 }; iter != end; ++iter) {
		filename_seg.push_back(iter->path().string());
	}

//	for (int i = 0; i < filename_seg.size(); i++)
//		cout << filename_seg[i] << endl;

	// Init Kalman parameters
	vector <KalmanFilter> kf_vector;
	for (int i = 0; i < no_frames; i++) {
		KalmanFilter KF(2, 2, 0);
		init_kalman(KF);
		kf_vector.push_back(KF); // state dim = 2, measurement dim = 2
	}

	// Loop through Loc_files and create vector
	vector <vector <int>> file_data, file_data1, file_data2;

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

	file_processing(filename_loc[0], file_data);
	vector <double> c;

	c.clear();
	for (int i = 0; i < file_data.size(); i++)
	{
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		c.push_back(255 * (rand() / (1.0 + RAND_MAX)));
		cout << "intial_color_allocation:" << c.at(0) << " " << c.at(1) << " " << c.at(2) << " " << endl;
		//cout << "intial_color_values:" << a << " " << b << " " << d << " " << endl;
		colors.push_back(c);
		c.clear();
	}
	
	for (int current_loc = 0; current_loc < no_frames - 1; current_loc++) 
	{
		//		if (tmp > 5) break;
		tmp++;
		cout << "Frame no: " << tmp << endl;

		// Create vector of center coordinates for current and next frame
		file_data1.clear(); // 2
		file_processing(filename_loc[current_loc + 1], file_data1);
		cout << "  Size of file_data: " << file_data.size() << endl;
		cout << "  Size of file_data1: " << file_data1.size() << endl;

		prediction = kf_vector[current_loc].predict();
		predictPt.x = prediction.at<float>(0);
		predictPt.y = prediction.at<float>(1);

		//convertFileToMat(filename_seg[current_loc], labelled0, binary0);
		//convertFileToMat(filename_seg[current_loc + 1], labelled, binary);
		binary0 = imread(filename_seg[current_loc], CV_LOAD_IMAGE_GRAYSCALE); //5
		binary = imread(filename_seg[current_loc + 1], CV_LOAD_IMAGE_GRAYSCALE); //6
		cvtColor(binary0, output0, CV_GRAY2BGR);
		cvtColor(binary, output, CV_GRAY2BGR);

		//operations(output); // if video is reversed (1/3)
		//operations(output0); // if video is reversed (2/3)

		for (int i = 0; i < file_data.size(); i++)
		{ // Loop over bats in 1 loc file
													 // cout << "  Bat no: " << i+1 << endl;

													 //			if (tmp1 > 5) break;
			tmp1++;
			measurement(0) = file_data[i][0];
			measurement(1) = file_data[i][1];
			current_bat.push_back(file_data[i][0]);
			current_bat.push_back(file_data[i][1]);
			//			cout << "measurement: " << measurement(0) << "," << measurement(1) << endl;

			estimation = kf_vector[current_loc].correct(measurement);
			estimated_bat.x = estimation.at<float>(0);
			estimated_bat.y = estimation.at<float>(1); // Estimation of KF
													   //			cout << "estimated: " << estimated_bat.x << "," << estimated_bat.y << endl;

			get_next_center(estimated_bat, predicted_bat, file_data1, index); // Likely next bat
																			  //			cout << "MF predicted Bat: " << predicted_bat[0] << "," << predicted_bat[1] << endl;
			file_data1.erase(file_data1.begin() + index);
			file_data2.push_back(predicted_bat);
			current_bat.clear();
			if (file_data1.size() == 0) {
				cout << "     Next frame has lesser bats!" << endl;
				break;
			}
		}

		file_data2.insert(std::end(file_data2), std::begin(file_data1), std::end(file_data1));
		//CvPoint* points = new CvPoint[2];
		cout << "  Size of file_data2: " << file_data2.size() << endl;
		color_bat(file_data, file_data2,colors, output, kf_vector);
		cv::namedWindow("output current frame", WINDOW_NORMAL);
		cv::imshow("output current frame", output0);
		cv::namedWindow("output next frame", WINDOW_NORMAL);
		cv::imshow("output next frame", output);
		file_data = file_data2;
		file_data2.clear();
		char key = waitKey(0);
	}

	return 0;
}

void operations(Mat &output) {
	//cv::rotate(output, output, ROTATE_180);
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
	//cout << single_pixel.at(0) <<single_pixel.at(1)<< single_pixel.at(2) << endl;

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
			//cout << single_pixel.at(0) << single_pixel.at(1) << single_pixel.at(2) << endl;
			line(output, Point(current_frame[i][0], current_frame[i][1]), Point(next_frame[i][0], next_frame[i][1]),
				Scalar(single_pixel.at(0), single_pixel.at(1), single_pixel.at(2)), 2, LINE_8);
			single_pixel.clear();
		}
		//catch (...)
		//{
		//	cout << "In catch" << i << endl;
		//	continue;
		//}
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
	


	//	Point center(predicted_bat[1], predicted_bat[0]); // if video is reversed (3/3)
	//circle(output, center, 3, Scalar(0, 0, 255), -1, 8);

	/*
	StackConnectedComponents(binary, predicted_bat, blob, src);
	cout << "Blob size: " << blob.size() << endl;
	for (int j = 0; j < blob.size(); j++) {
	Vec3b color;
	color[0] = 0;
	color[1] = 0;
	color[2] = 255;
	output.at<Vec3b>(Point(blob[j].y, blob[j].x)) = color;
	//cout << "binary values: " << (int)binary.at<uchar>(blob[j].y, blob[j].x) << endl;
	}
	//cout << "binary values zeros: " << (int)binary.at<uchar>(0, 0) << endl;
	*/
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
		//		cout << "i=" << i << "nfp=" << nfp.x << "," << nfp.y << " ";
		if (i == 0) {
			difference = sqrt(pow((estimated_bat.x - nfp.x), 2) + pow((estimated_bat.y - nfp.y), 2));
			index = 0;
			// compute distance between bats
		}
		else {
			distance = sqrt(pow((estimated_bat.x - nfp.x), 2) + pow((estimated_bat.y - nfp.y), 2));
			if (distance < difference) { // check for shortest distance
				difference = distance; // store value, index
				index = i;
			}
		}
		//		cout << "difference=" << difference << endl;
	}
	//cout << "index" << index << endl;
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

/*
for (int i = 0; i < file_data1.size(); i++)
cout << file_data1[i][0] << " " << file_data1[i][1] << endl;
//output0 = Mat::zeros(binary0.size(), CV_8UC3);
//output = Mat::zeros(binary.size(), CV_8UC3);
//namedWindow("binary current image", WINDOW_NORMAL);
//imshow("binary current image", binary0);
//namedWindow("binary next image", WINDOW_NORMAL);
//imshow("binary next image", binary);
*/
