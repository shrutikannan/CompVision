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

using namespace cv;
using namespace std;
//
void convertFileToMat(String filename, Mat& labelled, Mat& binary);
void file_processing(string filename_det, vector <vector<int>>& data_store);
void color_bat(vector <int> predicted_bat, string file_name, Mat& labelled, Mat& binary, Mat& output, std::vector<cv::Point> &blob, Mat1b &src);
void StackConnectedComponents(Mat &binary, vector <int> predicted_bat, std::vector<cv::Point>& blob, Mat1b &src);

////float A[4][4] = { { 1,0,1,0 },{ 0,1,0,1 },{ 0,0,1,0 },{ 0,0,0,1 } }; // Transition matrix

int main()
{
	vector <vector <int>> file_data,file_data1;
	int size_file;
	String filename_seg = "Segmentation_Bats/CS585Bats-Segmentation_frame000000751.txt";
	String filename_det = "Localization_Bats/CS585Bats-Localization_frame000000750.txt";
	String filename_det1 = "Localization_Bats/CS585Bats-Localization_frame000000751.txt";
	Mat labelled, binary;

	file_processing(filename_det,file_data);
	file_processing(filename_det1, file_data1);

	size_file = file_data.size();
	cout << "size of file:" << size_file << endl;

	KalmanFilter KF(2, 2, 0);//state_dim = 2, 
	setIdentity(KF.transitionMatrix);
	setIdentity(KF.measurementMatrix);//H
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R
	setIdentity(KF.errorCovPost, Scalar::all(1));//P
	KF.statePre.at<float>(0) = 0;
	KF.statePre.at<float>(1) = 0;

	Mat prediction = KF.predict();
	Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

	vector<int> value = file_data[0];
	Mat_<float> measurement(2, 1);
	measurement(0) = file_data[0][0];
	measurement(1) = file_data[0][1];
	cout << "file_data" << measurement(0)<< measurement(1) << endl;

	Mat Estimated = KF.correct(measurement);
	Point est(Estimated.at<float>(0), Estimated.at<float>(1));
	cout << "estimated" << Estimated << endl;
	
	float diff,d=0;
	int index = 0;
	for (int i = 0; i < file_data1.size(); i++)
	{
		Point nfp(file_data1[i][0], file_data1[i][1]);
		if (i == 0)
		{
			diff = sqrt((est.x - nfp.x) ^ 2 + (est.y - nfp.y) ^ 2);
		}
		else
		{
			d = sqrt((est.x - nfp.x) ^ 2 + (est.y - nfp.y) ^ 2);
			if (d < diff)
			{
				diff = d;
				index = i;
			}

		}
	}

	cout << "index" << index << endl;
	vector <int> predicted_bat = file_data1[index];
	std::vector<cv::Point> blob;

	convertFileToMat(filename_seg, labelled, binary);
	Mat output = Mat::zeros(binary.size(), CV_8UC3);
	//Mat output;
	//cvtColor(binary, output, CV_GRAY2BGR);
	Mat1b src = binary > 0;
	
	color_bat(predicted_bat , filename_seg, labelled, binary, output, blob, src);

	cout << "YES!!!!!!!MF Bat" << predicted_bat[0] << "," << predicted_bat[1] << endl;
	imshow("src", src);
	//for (int row = 0; row < binary.rows; row++) {			
	//	for (int col = 0; col < binary.cols; col++) {
	//		int value = (int)binary.at<uchar>(row, col);
	//		if (value!=0)
	//		{
	//			cout << "row:" << row << "col:" << col;	
	//		}
	//	}
	//	cout << endl;
	//}
	namedWindow("binary_image", WINDOW_NORMAL);
	imshow("binary_image", binary);
	rotate(output, output, ROTATE_180);
	transpose(output, output);
	flip(output,output, 0);
	flip(output,output, 1);
	namedWindow("output_image", WINDOW_NORMAL);
	imshow("output_image", output);
	char key = waitKey(0);
	return 0;
}

void file_processing(string filename_det, vector <vector<int>>& data_store)
{
	//cout << "1a" << endl;
	ifstream infile(filename_det);
	//vector <vector <int>> data_store;
	if (!infile)
	{
		cout << "Error reading file!";
		return;
	}
	
	//read the comma separated values into a vector of vector of ints
	while (infile)
	{
		string s;
		if (!getline(infile, s)) break;
	
		istringstream ss(s);
		vector <int> datarow;
	
		while (ss)
		{
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

void color_bat(vector <int> predicted_bat, string file_name, Mat& labelled, Mat& binary, Mat& output, std::vector<cv::Point> &blob, Mat1b &src)
{

	StackConnectedComponents(binary, predicted_bat, blob, src);
	cout << blob.size() << endl;

	for (int j = 0; j < blob.size(); j++) 
	{
				Vec3b color = output.at<Vec3b>(Point(blob[j].y,blob[j].x));

				color[0] = 0;
				color[1] = 0;
				color[2] = 255;

				output.at<Vec3b>(Point(blob[j].y, blob[j].x)) = color;
				cout << "binary values: " << (int)binary.at<uchar>(blob[j].y, blob[j].x) << endl;
	}
	cout << "binary values zeros: " << (int)binary.at<uchar>(0,0) << endl;
	
}

void StackConnectedComponents(Mat &binary, vector <int> predicted_bat, std::vector<cv::Point> &blob, Mat1b &src) { // Ref [3]
	//Mat1b src = binary > 0; // Boolean Matrix Ref [1]
	//cout << "start src" << (int)binary.at<uchar>(872, 413) << endl;
	int label = 0;
	int w = src.cols;
	int h = src.rows;
	int i;

	cv::Point point;								 //vector <Point2i> blob;
	int row = predicted_bat[1]; // changed 
	int col = predicted_bat[0];
//	int row = 863;
//	int col = 1019;
	std::stack<int, std::vector<int>> stack2; // Declare a stack
	i = col + row*w; // Information about the current row and col is here
	stack2.push(i); // Store in the stack
	//std::vector<cv::Point> blob; // Points belonging to 1 object stored here
	//cout << "Stack size" << stack2.size() << endl;
	while (!stack2.empty()) {// While the stack is not empty this loop will go on
		i = stack2.top(); // Extract the first element
		stack2.pop(); // Remove the first element
		
		int x2 = i%w; // Col number of current elmt
		int y2 = i / w; // Row no of current elmt
		src(y2, x2) = 0; // Set to 0, so that they are not checked again
		point.x = x2; // Creating a new point // exchange
		point.y = y2;
		//cout << "Point" << point << endl;
		blob.push_back(point); // Storing that point in a vector

		//cout << src(y2 - 1, x2 - 1) << src(y2 + 1, x2 - 1) << src(y2 - 1, x2 + 1) << src(y2 + 1, x2 + 1) << endl;

		if (x2 > 0 && y2 > 0 && (src(y2 - 1, x2 - 1) > 0)) { // Checking whether neighbours are object // changed
			//cout << "1" << endl;
			stack2.push(i - w - 1); // If yes, then pusheed to stack so that they can be analysed
			src(y2 - 1, x2 - 1) = 0; // Storing as 0, so that they are not checked again
		}
		if (x2 > 0 && y2 < h - 1 && (src(y2 + 1, x2 - 1) > 0)) {
			//cout << "2" << endl;
			stack2.push(i + w - 1);
			src(y2 + 1, x2 - 1) = 0;
		}
		if (x2 < w - 1 && y2>0 && (src(y2 - 1, x2 + 1) > 0)) {
			//cout << "3" << endl;
			stack2.push(i - w + 1);
			src(y2 - 1, x2 + 1) = 0;
		}
		if (x2 < w - 1 && y2 < h - 1 && (src(y2 + 1, x2 + 1) > 0)) {
			//cout << "4" << endl;
			stack2.push(i + w + 1);
			src(y2 + 1, x2 + 1) = 0;
		}
		cout << "stack size at end of while" << stack2.size() << endl;
	}		
	return;
}
//
void convertFileToMat(String filename, Mat& labelled, Mat& binary)
{
	//read file
	ifstream infile(filename);
	vector <vector <int>> data;
	if (!infile)
	{
		cout << "Error reading file!";
		return;
	}

	//read the comma separated values into a vector of vector of ints
	while (infile)
	{
		string s;
		if (!getline(infile, s)) break;

		istringstream ss(s);
		vector <int> datarow;

		while (ss)
		{
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
		for (int j = 0; j < labelled.cols; ++j)
		{
			binary.at<uchar>(i, j) = (labelled.at<uchar>(i, j) == 0) ? 0 : 255;
			//int pixel_value = (int)binary.at<uchar>(i, j);
				//cout << " " << pixel_value;
		}
		//cout << endl;
	}


}
//
///* FOR COLOURING
//Mat output = Mat::zeros(img.size(), CV_8UC3); // Colouring the different parts obtained by Component labelling
//for (size_t i = 0; i < blobs.size(); i++) { // From Lab
//unsigned char r = 255 * (rand() / 1.0 + RAND_MAX);
//unsigned char g = 255 * (rand() / 1.0 + RAND_MAX);
//unsigned char b = 255 * (rand() / 1.0 + RAND_MAX);
//for (size_t j = 0; j < blobs[i].size(); j++) {
//int x = blobs[i][j].x;
//int y = blobs[i][j].y;
//output.at<Vec3b>(y, x)[0] = b;
//output.at<Vec3b>(y, x)[1] = g;
//output.at<Vec3b>(y, x)[2] = r;
//}
//}
//cv::imshow("Labelled image", output);
//*/
//
//
//
////
////void drawObjectDetections(String filename, Mat& binary3channel)
////{
////	ifstream infile(filename);
////	vector <vector <int>> data;
////	if (!infile) {
////		cout << "Error reading file!";
////		return;
////	}
////
////	//read the comma separated values into a vector of vector of ints
////	while (infile)
////	{
////		string s;
////		if (!getline(infile, s)) break;
////
////		istringstream ss(s);
////		vector <int> datarow;
////
////		while (ss)
////		{
////			string srow;
////			int sint;
////			if (!getline(ss, srow, ',')) break;
////			sint = atoi(srow.c_str()); //convert string to int
////			datarow.push_back(sint);
////		}
////		data.push_back(datarow);
////	}
////
////
////	//draw red circles on the image
////	for (int i = 0; i < data.size(); ++i)
////	{
////		Point center(data.at(i).at(0), data.at(i).at(1));
////
////		circle(binary3channel, center, 3, Scalar(0, 0, 255), -1, 8);
////	}
////
////
////
////}
////*/// Kalman filter variables
////float A = 1;
////float Q = 0.1;
////float H = 1;
////float R = 0.5;
////float P_init = R;
////float R[2][2] = { {0.2845,0.0045},{0.0045,0.0045} }; // Measurement noise cov mat
////float H[2][4] = { { 1,0,0,0 },{ 0,1,0,0 } }; // Measurement mat
////float Q[4][4] = { {0.01,0,0,0}, {0,0.01,0,0}, {0,0,0.01,0}, {0,0,0,0.01} }; // pProcess noise cov mat
////float P[4][4] = { { 100,0,0,0 },{ 0,100,0,0 },{ 0,0,100,0 },{ 0,0,0,100 } }; // Covariance matrix
////int dt = 1;