/*
Author: Shruti Kannan
Title: Assignment 03
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>
#include <iomanip>
#include <stack>

using namespace cv;
using namespace std;
Mat3b out;
Mat negate_pixel(Mat src);
//void RecursiveConnectedComponents(Mat img_negate, int row, int col, int label);
void StackConnectedComponents(Mat img_binary, Mat1i labels, Mat img_labelled);
void show_components(Mat img_negate);
void img_erosion(Mat src, Mat dst);

vector<Mat> neighbour_list;
//void applyCustomColormap(const Mat1i& src, Mat3b& dst);
Mat output;// = Mat::zeros(img.size(), CV_8UC3);

vector <vector<Point2i>> blobs; //initialize vector to store all blobs


int main()
{
	/*
	StackConnectedComponents(label)
	declare a stack data structure
	push the first region pixel into the stack

	while (the stack is not empty){
	pop an item off the stack
	for each unmarked neighbor{
	assign the label to the neighbor
	push the neighbor into the stack
	}
	}
	*/
	Mat img = imread("open_fist-bw.png", IMREAD_GRAYSCALE);
	imshow("image", img);

	Mat img_negate = negate_pixel(img);
	imshow("Negated image", img_negate);
	//	show_components(img_negate);
	Mat img_binary;
	threshold(img_negate, img_binary, 0.0, 1.0, THRESH_BINARY);
	imshow("Binary image", img_binary);
	
	Mat1i labels;
	Mat img_labelled = img_binary.clone();
	StackConnectedComponents(img_binary, labels, img_labelled);
	
	Mat output = Mat::zeros(img.size(), CV_8UC3);
	for (size_t i = 0; i < blobs.size(); i++) { 	// Randomly color the blobs
		unsigned char r = 255 * (rand() / 1.0 + RAND_MAX);
		unsigned char g = 255 * (rand() / 1.0 + RAND_MAX);
		unsigned char b = 255 * (rand() / 1.0 + RAND_MAX);
		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			output.at<Vec3b>(y, x)[0] = b;
			output.at<Vec3b>(y, x)[1] = g;
			output.at<Vec3b>(y, x)[2] = r;
		}
	}

	imshow("labelled", output);
	waitKey(0);
	return 0;
}
/*
void RecursiveConnectedComponents(Mat src, int row, int col, int label) {
//	RecursiveConnectedComponents(pixel, label)
//			for each unmarked neighbor
//			assign the label to the neighbor
//			call RecursiveConnectedComponents(neighbor, label)
	
	src.at<uchar>(row, col) = label;// assign label to current pixel
	for (int dx = -1; dx <= 1; dx++) { // loop over neighbours
		int *rowPtr = (int*)src.ptr(row);
		for (int dy = -1; dy <= 1; dy++) {
			if (rowPtr[col] == 1) { // for each neighbour pixel, check if label is -1 (part of object)
				RecursiveConnectedComponents(src, row + dx, col + dy, label); // if yes, then change label for that too, and look among its neighbours
			}
		}
	}
	return;
}
*/

/*
StackConnectedComponents(label)
declare a stack data structure
push the first region pixel into the stack

while (the stack is not empty) {
	pop an item off the stack
		for each unmarked neighbor{
			assign the label to the neighbor
			push the neighbor into the stack
		}
}*/

void StackConnectedComponents(Mat img_binary, Mat1i labels, Mat img_labelled) {
	Mat1b src = img_binary > 0;
	labels = Mat1i(img_binary.rows, img_binary.cols, 0);

	int label = 0;
	int w = src.cols;
	int h = src.rows;
	int i;

	cv::Point point;
	for (int row = 0; row<h; row++) {
		int *rowP = (int*)labels.ptr(row);
		for (int col = 0; col<w; col++) {
			if ((src(row, col)) > 0) {   // Non zero element
//			if (rowP[col] > 0) {
				vector <Point2i> blob;
				std::stack<int, std::vector<int>> stack2; // Declare a stack
				i = col + row*w; // information about the current row and col is here
				stack2.push(i); // store in the stack

				std::vector<cv::Point> comp; // points belonging to 1 object stored here

				while (!stack2.empty()) // while the stack is not empty this loop will go on
				{
					i = stack2.top(); // extract the first element
					stack2.pop(); // remove the first element

					int x2 = i%w; // col number of current elmt
					int y2 = i / w; // row no of current elmt

//					src(y2, x2) = 0; // set to 0
					

					point.x = x2; // creating a new point
					point.y = y2;
					comp.push_back(point); // storing that point in a vector

					if (x2 > 0 && y2 > 0 && (src(y2 - 1, x2 - 1) != 0))
					{
						stack2.push(i - w - 1);
						src(y2 - 1, x2 - 1) = 0;
					}
					if (x2 > 0 && y2 < h - 1 && (src(y2 + 1, x2 - 1) != 0))
					{
						stack2.push(i + w - 1);
						src(y2 + 1, x2 - 1) = 0;
					}
					if (x2 < w - 1 && y2>0 && (src(y2 - 1, x2 + 1) != 0))
					{
						stack2.push(i - w + 1);
						src(y2 - 1, x2 + 1) = 0;
					}
					if (x2 < w - 1 && y2 < h - 1 && (src(y2 + 1, x2 + 1) != 0))
					{
						stack2.push(i + w + 1);
						src(y2 + 1, x2 + 1) = 0;
					}
				}
				blobs.push_back(comp);
				++label;
			}
		}
	}
	return;
}

Mat negate_pixel(Mat src) {
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < 50) {
				//			if(src.at<uchar>(i,j) == 1) {
				dst.at<uchar>(i, j) = -1;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

void show_components(Mat img_negate) {
	for (int row = 0; row < img_negate.rows; row++) {
		for (int col = 0; col < img_negate.cols; col++) {
			cout << (int)img_negate.at<uchar>(row, col);
		}
	}
}
/*
void img_erosion(Mat src, Mat dst) {
	for (int row = 0;row < src.rows;row++) {
		int *rowP = (int*)src.ptr(row);
		for (int col = 0; col < src.col-1;col++) {
			if (rowP[col] && rowP[col+1] && )
		}
	}
}*/
