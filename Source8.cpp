/*
Author : G Siva Perumal
The code does the following:
1. Takes the input frames and crops the area of interest
2. Uses thresholding to get rid of the piano and have only the hands
3. Erosion to remove noise
4. Finds contours corresponding to the hands
5. Draws a Rectangle on the hands and creates an output video
*/

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <Windows.h>

using namespace std;
using namespace cv;

void find_contours(Mat diff, Mat original);

int main()
{
	std::string location;
	vector<string> loc;

	string path = "CS585-PeopleImages/frame_0";
	string num;

	//creating filenames and putting them in the vector
	for (int i = 10; i <= 160; i++) {
		if (i >= 10 && i < 100) {
			num = "0" + to_string(i);
		}
		else {
			num = to_string(i);
		}
		location = path + num + ".jpg";
		loc.push_back(location);
	}
	cout << loc.size() << endl;

	//read each image
//	for (int i = 1; i < 2; i++) //loc.size() - 1; i++)
//	{
		Mat img, img_prev;
		img = imread(loc[0]);
		img_prev = imread(loc[1]);
		if (!img.data || !img_prev.data) {
			cout << "Image not loaded" << endl;
			return -1;
		}
		
		Mat diff;
		diff = img_prev - img;

		cvtColor(diff, diff, CV_RGB2GRAY);
		imshow("Diff", diff);
		int erosion_size = 1;
		Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
		erode(diff, diff, element, Point(-1, -1), 1);
		dilate(diff, diff, element, Point(-1, -1), 1);
		imshow("Diff_eroded", diff);
		//waitKey(0);
		//Mat diff_binary = Mat::zeros(diff.size(), CV_8UC1);
		//for (int i = 0; i < diff.rows; i++) {
		//	int* rowP = (int*)diff.ptr(i);
		//	int* rowP1 = (int*)diff_binary.ptr(i);
		//	for (int j = 0;j < diff.cols;j++) {
		//		if (diff.at<uchar>(i,j) > 50) {
		//			//cout << "white pixel" << " " << i<< " " << j << endl;
		//			rowP1[j] = 255;
		//		}
		//	}
		//}
	//	waitKey(0);
//	}
		find_contours(diff, img);
	waitKey(0);
	return 0;
}

void find_contours(Mat diff, Mat original) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//find contours in the src image
	findContours(diff, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat contour_output = Mat::zeros(diff.size(), CV_8UC3);
	cout << "The number of contours detected is: " << contours.size() << endl;
	return;
}
