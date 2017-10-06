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

//function that removes the piano by thresholding and detects the skin
void remove_piano(Mat& src, Mat& dst);

//calculates the min amongst three integers
int myMin(int a, int b, int c);

//calculates the max amongst three integers
int myMax(int a, int b, int c);

//compares the areas of two contours
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);

//finds the hands and puts a rectangle on the hand
Mat find_contour_hand(Mat& src, Mat& original);


int main()
{
	std::string location;
	vector<string> loc;

	string path = "CS585-PianoImages/piano_";

	//creating filenames and putting them in the vector
	for (int i = 14; i <= 35; i++)
	{
		//these file numbers are not present
		if (i ==20 || i ==21 || (i >= 28 && i <= 32))
		{

			continue;

	}
		location = path + to_string(i) + ".png";
		loc.push_back(location);
		
	}

	//from the image crop the important part with this rectangle
	Rect crop_area(1022, 400, 220, 830);

	cout << loc.size() << endl;

	//VideoWriter to create output video
	VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'),3, Size(220, 830));


	//read each image
	for (int i = 0; i <loc.size()-1; i++)
	{	
		Mat img;
		img = imread(loc[i]);
		if (!img.data)
		{
			cout << "Image not loaded" << endl;
			return -1;
		}
		//crop the image 
		Mat cropped_image = img(crop_area);

		Mat dst;// create a mat for piano removed, skin detected image
		dst = Mat::zeros(cropped_image.rows, cropped_image.cols, CV_8UC1);
		remove_piano(cropped_image, dst);
		
		//erode the piano removed, skin detected image to remove noise
		Mat erodedst;
		erodedst = Mat::zeros(cropped_image.rows, cropped_image.cols, CV_8UC1);
		int erosion_size = 1;
		Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
		erode(dst, erodedst, element, Point(-1, -1), 1);

		// give the eroded image to find_contour_hand to find contours of the hand
		Mat copy = cropped_image.clone();
		Mat final_output =find_contour_hand(erodedst, copy);

		//write the output
		video.write(final_output);


		
	}
	video.release();
	return 0;

}

int myMax(int a, int b, int c)
{
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

void remove_piano(Mat & src, Mat & dst)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{

			Vec3b intensity = src.at<Vec3b>(i, j);

			int B = intensity[0];
			int G = intensity[1];
			int R = intensity[2];

			if ((R > 100 && G > 115 && B > 115))  // Threshold values to remove piano
			{
				dst.at<uchar>(i, j) = 0;
			}
			else
			{
				//threshold to detect skin
				if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B))
					dst.at<uchar>(i, j) = 1;
			}
		}
	}
}

Mat find_contour_hand(Mat& src, Mat& original)

{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//find contours in the src image
	findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat contour_output = Mat::zeros(src.size(), CV_8UC3);
	cout << "The number of contours detected is: " << contours.size() << endl;

	//sort the contours according to their areas
	sort(contours.begin(), contours.end(), compareContourAreas);

	//find the biggest two contours
	vector<Point> biggestContour = contours[contours.size() - 1];
	vector<Point> biggestContour1 = contours[contours.size() - 2];

	
	//if two hands are together just draw one rectangle. We can find if two hands are together by finding the area of the biggest contour
	if (fabs(contourArea(cv::Mat(biggestContour))) > 3000)
	{
		Rect boundrec = boundingRect(biggestContour);
		rectangle(original, boundrec, Scalar(0, 255, 0), 1, 8, 0);
	}

	//if the two hands are not together draw two rectangles
	else
	{
		Rect boundrec = boundingRect(biggestContour);
		Rect boundrec1 = boundingRect(biggestContour1);
		rectangle(original, boundrec, Scalar(0, 255, 0), 1, 8, 0);
		rectangle(original, boundrec1, Scalar(255, 0, 0), 1, 8, 0);
	}
	namedWindow("original", CV_WINDOW_AUTOSIZE);
	imshow("original", original);
	// return the output frame to be included in the video

	return original;
}

//ref : https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}


	