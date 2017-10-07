//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <stack>
#include <stdlib.h>
#include <fstream>


using namespace cv;
using namespace std;

Mat img, img1, adap_img1;
Mat th1, color_th1;
vector<Mat> spl, spl2;
vector<Mat> vecValMean, vecValGaus;
Mat out_v, out_h, out_s;
Mat aout_v_mean, aout_v_gaus, aout_h_mean, aout_h_gaus, aout_s_mean, aout_s_gaus;
int max_thresh = 255;


int fr_number = 0;
//=============================================
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);
//void getStats(Mat &src);
int main(int argc, char** argv)
{
	Mat frame0 = imread("./CS585-PeopleImages/frame_0010.jpg");

	int initial = 11;
	int final = 160;
	int iter = final - (initial)+1;
	//int iter = 5;
	//VideoWriter vid("bat_orig.avi",CV_FOURCC('D','I','V','X'),20,Size(1024,1024),true);

	for (int i = 0; i < iter; i++) {
		string file1;
		int folded = 0;
		int spreadOut = 0;
		string ini = "./CS585-PeopleImages/frame_0";
		cout << "OK" << endl;

		string fin = ".jpg";
		string num1 = static_cast<ostringstream*>(&(ostringstream() << (initial + i)))->str();

		if (num1.size() == 2) num1 = "0" + num1;
		else if (num1.size() == 3) num1 = num1;

		cout << num1 << " " << endl;
		file1.append(ini); file1.append(num1); file1.append(fin);

		cout << file1 << " " << endl;

		Mat frame = imread(file1, 1);
		Mat frameDest;
		frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		myFrameDifferencing(frame0, frame, frameDest);


		adaptiveThreshold(frameDest, frameDest, max_thresh, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, -15);

		int erosion_size = 1;
		int dilation_size = 1;
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
		//erode(frameDest, frameDest, element);
		dilate(frameDest, frameDest, element);
		dilate(frameDest, frameDest, element);
		vector<vector<Point> > contour;
		vector<Vec4i> hierarchy;
		Mat copy_src = frameDest.clone();
		findContours(copy_src, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		//getStats(adap_img1);
		cout << contour.size() << endl;
		int people = 0;
		if (contour.size() > 0) {
			for (int i = 0; i < contour.size(); i++)
			{
				double area = contourArea(contour[i]);
				double perimeter = arcLength(contour[i], true);
				if (area > 500) {
					people++;
					cout << i << "   Area   " << area << endl;
					Rect boundrec = boundingRect(contour[i]);
					rectangle(frameDest, boundrec, Scalar(255, 0, 0), 1, 8, 0);
				}
			}
		}
		cout << "People: "<< people << endl;
		frame0 = frame;
		//imshow("Bats", adap_img1);
		imshow("Pedestrian", frameDest);
		waitKey(30);
	}
	//output_file2.close();

	waitKey(0);
	return 0;
}
// ./CS585-PeopleImages/frame_0010.jpg

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
	//For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
	//For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
	absdiff(prev, curr, dst);
	Mat gs = dst.clone();
	cvtColor(dst, gs, CV_BGR2GRAY);
	dst = gs > 50;
}