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
vector<Mat> neighbour_list;
void applyCustomColormap(const Mat1i& src, Mat3b& dst);
Mat output;// = Mat::zeros(img.size(), CV_8UC3);
void color(Mat1i labels, Mat output, int label);

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
	
//	cout << "_____________"<<labels;
//	Mat3b out;
//	applyCustomColormap(labels, out);
//	show_components(out);
	imshow("Labels", out);
//	cout << output;
//	imshow("Labels", output);
	
	waitKey(0);
	return 0;
}



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
				++label;
				for (int k = 0; k <comp.size(); ++k)
				{
					labels(comp[k]) = label;
	//				int *rowP = (int*)img_labelled.ptr(comp[k].x);
	//				rowP[comp[k].y] = label;
					//cout << comp[k].x;
					//img_labelled.at<uchar>(comp[k].x, comp[k].y) = label;
					
				}
				cout << labels.size();
			}
		}
	}
	applyCustomColormap(labels, out);
//	applyColorMap(labels, out, COLORMAP_JET);
//	color(labels, output, label);
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
void color(Mat1i labels, Mat output, int label) {
	
	for (size_t i = 0; i < label - 1; i++) {
		unsigned char r = 255 * (rand() / 1.0 + RAND_MAX);
		unsigned char g = 255 * (rand() / 1.0 + RAND_MAX);
		unsigned char b = 255 * (rand() / 1.0 + RAND_MAX);
		for (int j = 1;j<labels.size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			output.at<Vec3b>(y, x)[0] = b;
			output.at<Vec3b>(y, x)[1] = g;
			output.at<Vec3b>(y, x)[2] = r;
			cout << output;
		}
		for (int row = 0; row < labels.rows - 1; row++) {
			int *rowP = (int*)labels.ptr(row);
			for (int col = 1; col < labels.cols - 1; col++) {
				cout << rowP[col];
			}
			cout << endl;
		}
	}
}*/


void applyCustomColormap(const Mat1i& src, Mat3b& dst) //https://stackoverflow.com/questions/35993895/create-a-rgb-image-from-pixel-labels/35995427#35995427
{
	// Create JET colormap

	double m;
	minMaxLoc(src, nullptr, &m);
	m++;

	int n = ceil(m / 4);
	Mat1d u(n * 3 - 1, 1, double(1.0));

	for (int i = 1; i <= n; ++i) {
		u(i - 1) = double(i) / n;
		u((n * 3 - 1) - i) = double(i) / n;
	}

	std::vector<double> g(n * 3 - 1, 1);
	std::vector<double> r(n * 3 - 1, 1);
	std::vector<double> b(n * 3 - 1, 1);
	for (int i = 0; i < g.size(); ++i)
	{
		g[i] = ceil(double(n) / 2) - (int(m) % 4 == 1 ? 1 : 0) + i + 1;
		r[i] = g[i] + n;
		b[i] = g[i] - n;
	}

	g.erase(std::remove_if(g.begin(), g.end(), [m](double v) { return v > m; }), g.end());
	r.erase(std::remove_if(r.begin(), r.end(), [m](double v) { return v > m; }), r.end());
	b.erase(std::remove_if(b.begin(), b.end(), [](double v) { return v < 1.0; }), b.end());

	Mat1d cmap(m, 3, double(0.0));
	for (int i = 0; i < r.size(); ++i) { cmap(int(r[i]) - 1, 0) = u(i); }
	for (int i = 0; i < g.size(); ++i) { cmap(int(g[i]) - 1, 1) = u(i); }
	for (int i = 0; i < b.size(); ++i) { cmap(int(b[i]) - 1, 2) = u(u.rows - b.size() + i); }

	Mat3d cmap3 = cmap.reshape(3);

	Mat3b colormap;
	cmap3.convertTo(colormap, CV_8U, 255.0);

	// Apply color mapping
	dst = Mat3b(src.rows, src.cols, Vec3b(0, 0, 0));
	for (int r = 0; r < src.rows; ++r)
	{
		for (int c = 0; c < src.cols; ++c)
		{
			dst(r, c) = colormap(src(r, c));
		}
	}
}

