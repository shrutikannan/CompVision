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
#include <string>

using namespace cv;
using namespace std;
Mat3b out;
Mat negate_pixel(Mat src);
//void RecursiveConnectedComponents(Mat img_negate, int row, int col, int label);
void StackConnectedComponents(Mat img_binary);
void show_components(Mat img_negate);
void img_erosion(Mat src, Mat dst);
void img_dilation(Mat src, Mat dst);
void img_boundary(Mat src, std::vector<Point>& BoundaryPoints);
vector<Mat> neighbour_list;
//void applyCustomColormap(const Mat1i& src, Mat3b& dst);
Mat output;// = Mat::zeros(img.size(), CV_8UC3);
void TraceBoundaryPoints(Mat InputImage,
	int Width_i, int Height_i,
	std::vector<Point2i>& BoundaryPoints);
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
	Mat img = imread("open-bw-full.png", IMREAD_GRAYSCALE);
	imshow("image", img);

	Mat img_negate1 = negate_pixel(img);
	imshow("Negated image", img_negate1);

	Mat img_negate = img_negate1.clone();
	img_erosion(img_negate1, img_negate);
	img_erosion(img_negate1, img_negate);
	img_erosion(img_negate1, img_negate);
	img_dilation(img_negate1, img_negate);
	img_dilation(img_negate1, img_negate);
	imshow("eroded", img_negate);
	
	Mat img_binary = Mat::zeros(img_negate.size(), CV_8UC1);
	threshold(img_negate, img_binary, 0.0, 1.0, THRESH_BINARY);
	imshow("Binary image", img_binary);

	StackConnectedComponents(img_binary);
	
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
	std::vector<Point> BoundaryPoints;
	img_boundary(img_binary, BoundaryPoints);
//	TraceBoundaryPoints(img, img_binary.rows, img_binary.cols, BoundaryPoints);
	Mat bound = Mat::zeros(img_binary.size(), CV_8UC1);
//	cout << BoundaryPoints.size();

	for (int i = 0; i < BoundaryPoints.size(); i++) {
		
		int x = BoundaryPoints[i].x;
		int y = BoundaryPoints[i].y;
		//cout << BoundaryPoints[i] << " " << x << " " << y << endl;
		//int* rowP = (int*)dst.ptr(x);
		bound.at<uchar>(x,y) = 255;
//		dst.at<Vec3b>(y, x)[0] = 0;
//		dst.at<Vec3b>(y, x)[1] = 255;
//		dst.at<Vec3b>(y, x)[2] = 0;
	}
	imshow("boundary",bound);
	waitKey(0);
	return 0;
}

void img_boundary(Mat src, std::vector<Point>& BoundaryPoints) {
	// 1st non zero pixel is start 's'
	// current pixel is 'p'
	// pixel before p is 'b' backtracked pixel
	// 3*3 neighbourhood of p, starting from b, search clockwise for the next non zero pixel
	// if found, that becomes the current pixel, and what p was, becomes the backtracked pixel

	int rows = src.rows;
	int cols = src.cols;
	int no_pixels = rows*cols;
	int neighborhood[16][2] = { {0,-1 },{-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{ 0,-1 },{ -1,-1 },{ -1,0 },{ -1,1 },{ 0,1 },{ 1,1 },{ 1,0 },{ 1,-1 } };
	BoundaryPoints.clear();
	cout << "1" << BoundaryPoints.size() << endl;;
	Point s, b, c, p;
	Point boundarypixel;
	int temp = 0;
	for (int row = 0; row < rows; row++) { // loop to find the starting point
		int* rowP = (int*)src.ptr(row);
		for (int col = 0; col < cols; col++) {
			if (rowP[col] != 0) { // 1st non zero element found
				s.x = row; // = col + row*rows; //starting boundary pixel
				s.y = col;
				b.x = row; //(col - 1) + row*rows; // backtrack pixel
				b.y = col - 1;
				BoundaryPoints.push_back(s); // storing s in the final boundary
				temp = 1; // since we have found the starting point s
				break;
			}
		}
		if (temp == 1) {
			temp = 0;
			break;
		}
	}
	cout << "2" << BoundaryPoints.size() << endl;;
	p = s; // current pixel to be analysed, p, is s
	cout << "s = "<< s <<endl;
	c.x = p.x + neighborhood[0][0]; // initialise c with the neighborhood pixel of s
	c.y = p.y + neighborhood[0][1];
	cout << "before while" << endl;
	vector<Point> c_values;
	int start_idx = 0;
	int end_idx = 8;
	while (c != s) { // boundary gets completed once c is equal to s
		//cout << "p= " << p << endl;
		cout << "3" << BoundaryPoints.size() << endl;

		for (int ind_nei = start_idx; ind_nei < end_idx; ind_nei++) { // loop around neighbours
			cout << "inside for" << endl;
			cout << "1 start= " << start_idx << " " << "end=" << end_idx << endl;
			c.x = p.x + neighborhood[ind_nei][0]; // initialise c with the neighborhood pixel of s
			c.y = p.y + neighborhood[ind_nei][1];
			cout << "c = " << c << endl;
			c_values.push_back(c);
			for (int i = 0;i < c_values.size();i++)
				cout << c_values[i];
			cout << endl;

			cout << "1 c_values = " << c_values.size() << endl;
			int* rowP = (int*)src.ptr(c.x);
			if (rowP[c.y] != 0) { // if neighboring pixel is also white
				BoundaryPoints.push_back(c);				
				p = c_values.back();
				cout << "2 c_values = " << c_values.size() << endl;
				cout << "updated p = " << p << endl;
				c_values.pop_back();
				cout << "3 c_values = " << c_values.size() << endl;
				b = c_values.back();
				cout << "updated b = " << b << endl;
				c_values.clear();
				Point offset;
				offset.x = b.x - p.x;
				offset.y = b.y - p.y;
				for (int idx = 0; idx < 8; idx++) {
					if (offset.x == neighborhood[idx][0] && offset.y == neighborhood[idx][1]) {
						start_idx = idx;
						end_idx = start_idx + 8;
						cout << "start= " << start_idx << " " << "end=" << end_idx <<endl;
						break;
					}
				}
				break; // p is updated, need to check new neighbours
			}
		}
		//cout << endl;
	}
	return;
}


void StackConnectedComponents(Mat img_binary) {
	Mat1b src = img_binary > 0;

	int label = 0;
	int w = src.cols;
	int h = src.rows;
	int i;

	cv::Point point;
	for (int row = 0; row<h; row++) {
		for (int col = 0; col<w; col++) {
			if ((src(row, col)) > 0) {   // Non zero element
				vector <Point2i> blob;
				std::stack<int, std::vector<int>> stack2; // Declare a stack
				i = col + row*w; // information about the current row and col is here
				stack2.push(i); // store in the stack
				std::vector<cv::Point> comp; // points belonging to 1 object stored here

				while (!stack2.empty()) {// while the stack is not empty this loop will go on
					i = stack2.top(); // extract the first element
					stack2.pop(); // remove the first element

					int x2 = i%w; // col number of current elmt
					int y2 = i / w; // row no of current elmt

					src(y2, x2) = 0; // set to 0
					point.x = x2; // creating a new point
					point.y = y2;
					comp.push_back(point); // storing that point in a vector

					if (x2 > 0 && y2 > 0 && (src(y2 - 1, x2 - 1) != 0)) {
						stack2.push(i - w - 1);
						src(y2 - 1, x2 - 1) = 0;
					}
					if (x2 > 0 && y2 < h - 1 && (src(y2 + 1, x2 - 1) != 0)) {
						stack2.push(i + w - 1);
						src(y2 + 1, x2 - 1) = 0;
					}
					if (x2 < w - 1 && y2>0 && (src(y2 - 1, x2 + 1) != 0)) {
						stack2.push(i - w + 1);
						src(y2 - 1, x2 + 1) = 0;
					}
					if (x2 < w - 1 && y2 < h - 1 && (src(y2 + 1, x2 + 1) != 0)) {
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

void img_erosion(Mat src, Mat dst) {
	src = dst.clone();
	dst = Mat::zeros(src.size(), CV_8UC1);
	int count=0;
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			for (int dx = -1; dx <= 1; dx++) {
				for (int dy = -1; dy <= 1; dy++) {
					if (src.at<uchar>(i + dx, j + dy) != 0) {
						count++;
					}
				}
			}		
		if (count == 9) dst.at<uchar>(i, j) = 255;
		count = 0;
		}
	} 
	return;
}

void img_dilation(Mat src, Mat dst) {
	src = dst.clone();
	dst = Mat::zeros(src.size(), CV_8UC1);
	int count = 0;
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) != 0) {
				for (int dx = 0; dx <= 1; dx++) {
					for (int dy = 0; dy <= 1; dy++) {
						dst.at<uchar>(i + dx, j + dy) = 255;
					}
				}
			}
		}
	}
	return;
}
/*
References:
https://www.researchgate.net/post/What_is_a_basic_difference_between_Mat_Mat3b_Matx_etc_if_any_left_in_OpenCV
http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html

*/
