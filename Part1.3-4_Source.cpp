/**
CS585_Lab4.cpp
//Author - Prateek Mehta
CS585 Image and Video Computing Fall 2017
Lab 4
--------------
This program introduces the following concepts:
a) Understanding and applying basic morphological operations on images
b) Finding and labeling blobs in a binary image
--------------
*/


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
Function that detects blobs from a binary image
@param binaryImg The source binary image (binary image contains pixels labeled 0 or 1 (not 255))
@param blobs Vector to store all blobs, each of which is stored as a vector of 2D x-y coordinates
*/
int thresh = 128;
int max_thresh = 255;

void FindBinaryLargeObjects(const Mat &binaryImg, vector <vector<Point2i>> &blobs);
void threshold_callback(int, void* x);
void skeleton(Mat img);

int main(int argc, char **argv)
{
	// read image as grayscale
	Mat image = imread("hand_black.jpg", 0);
	Mat img = cv::Scalar::all(255) - image;
	if (!img.data) {
		cout << "File not found" << std::endl;
		return -1;
	}

	// create windows
	namedWindow("Original");
	namedWindow("binary");
	namedWindow("labelled");
	imshow("Original", img);
	// perform morphological operations to remove small objects (noise) and fill black spots inside objects
	// %TODO
	// Documentation on erode: http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=erode#erode
	// Documentation on dilate: http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=erode#dilate

	// initialize structuring element for morphological operations
	int erosion_size = 3;
	int dilation_size = 3;
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	skeleton(img);
	//perform erosions and dilations
	//erode(img, img, element);
	//erode(img, img, element);

	//dilate(img, img, element);
	//erode(img, img, element);
	//dilate(img, img, element);

	

	createTrackbar("Threshold:", "Original", &thresh, max_thresh, threshold_callback, &img);
	cout << "OK" << endl;
	//convert thresholded image to binary image
	Mat binary;
	threshold(img, binary, 0.0, 1.0, THRESH_BINARY);

	//initialize vector to store all blobs
	vector <vector<Point2i>> blobs;

	//call function that detects blobs and modifies the input binary image to store blob info
	//FindBinaryLargeObjects(binary, blobs);

	//display the output
	Mat output = Mat::zeros(img.size(), CV_8UC3);
	// Randomly color the blobs
	for (size_t i = 0; i < blobs.size(); i++) {
		unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));

		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			output.at<Vec3b>(y, x)[0] = b;
			output.at<Vec3b>(y, x)[1] = g;
			output.at<Vec3b>(y, x)[2] = r;
		}
	}

	//show the binary image, as well as the labelled image
	imshow("binary", img);
	//imshow("labelled", output);
	waitKey(0);

	return 0;
}

void FindBinaryLargeObjects(const Mat &binary, vector <vector<Point2i>> &blobs)
{
	//clear blobs vector
	blobs.clear();

	//labelled image of type CV_32SC1
	Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	//label count starts at 2
	int label_count = 2;

	//iterate over all pixels until a pixel with a 1-value is encountered
	for (int y = 0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for (int x = 0; x < label_image.cols; x++) {
			if (row[x] != 1) {
				continue;
			}
			cout << x<<endl;
			//floodFill the connected component with the label count
			//floodFill documentation: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
			Rect rect;
			floodFill(label_image, Point(x, y), label_count, &rect, 0, 0, 4);

			//store all 2D co-ordinates in a vector of 2d points called blob
			vector <Point2i> blob;
			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if (row2[j] != label_count) {
						continue;
					}
					blob.push_back(Point2i(j, i));
				}
			}
			//store the blob in the vector of blobs
			blobs.push_back(blob);

			//increment counter
			label_count++;
			cout << blobs.size() << endl;

			
		}
	}
	//cout << "The number of blobs in the image is: " << label_count;
	//Code derived from: http://nghiaho.com/
}

void threshold_callback(int, void* x)
{
	Mat thres_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	cout << "Threshold: " << thresh << endl;
	// Convert into binary image using thresholding
	// Documentation for threshold: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
	// Example of thresholding: http://docs.opencv.org/doc/tutorials/imgproc/threshold/threshold.html
	threshold(*(Mat *)x, thres_output, thresh, max_thresh, 0); // x is cast as a pointer to a matrix (Mat) and then we dereference the pointer

															   // Create Window and display thresholded image
	namedWindow("Thres", CV_WINDOW_AUTOSIZE);
	//imshow("Thres", thres_output);

	// Find contours
	// Documentation for finding contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(thres_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat contour_output = Mat::zeros(thres_output.size(), CV_8UC3);
	//cout << "The number of contours detected is: " << contours.size() << endl;

	// Find largest contour
	int maxsize = 0;
	int maxind = 0;
	Rect boundrec;

	if (contours.size() > 0) {
		for (int i = 0; i < contours.size(); i++)
		{
			// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
			double area = contourArea(contours[i]);
			double perimeter = arcLength(contours[i], true);
			double circularity = (perimeter*perimeter) / (4 * 3.14 * area);
			if (contourArea(contours[i]) > 1000) {
				std::cout << " Area of Objects: " << contourArea(contours[i]) << std::endl;
				std::cout << " Perimeter of Objects: " << arcLength(contours[i], true) << std::endl;
				std::cout << " Circularity of Objects: " << circularity << std::endl;
				std::cout << " Compactness of Objects: " << contourArea(contours[i])/ arcLength(contours[i], true) << std::endl;
				
			}

			if (area > maxsize) {
				maxsize = area;
				maxind = i;
				boundrec = boundingRect(contours[i]);
			}
		}

		// Draw contours
		// Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
		//drawContours(contour_output, contours, maxind, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
		//cout << "Done" << endl;
		drawContours(contour_output, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);
		// Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
		//rectangle(contour_output, boundrec, Scalar(0, 255, 0), 1, 8, 0);

		Moments m = cv::moments(contours[maxind], true);
		double cen_x = m.m10 / m.m00; //Centers are right
		double cen_y = m.m01 / m.m00;

		//double a = m.m20 / m.m00 - m.m00*cen_x*cen_x;
		//double b = 2 * m.m11 / m.m00 - m.m00*(cen_x*cen_x + cen_y*cen_y);
		//double c = m.m02 / m.m00 - m.m00*cen_y*cen_y;

		double a = m.m20;
		double b = m.m11;
		double c = m.m02;

		double theta = atan2(b, a - c) / 2.0;
		cout << "Thetaaaa: " << theta*180/3.14 << endl;


		RotatedRect calculatedRect = minAreaRect(contours[maxind]);
		if (calculatedRect.size.width<calculatedRect.size.height) {
			calculatedRect.angle = 90 - calculatedRect.angle;
		}
		else {
			calculatedRect.angle = -calculatedRect.angle;
		}
		//cout << "-----------------------------" << endl << endl;
		cout << "The orientation of the largest contour detected is: " << calculatedRect.angle << endl;
		cout << "-----------------------------" << endl << endl;

		Point3d p = Point3d(cen_x, cen_y, theta);
		line(thres_output, (Point(p.x, p.y) - Point(100 * cos(p.z), 100 * sin(p.z))), (Point(p.x, p.y) + Point(100 * cos(p.z), 100 * sin(p.z))), Scalar(0.5), 1);
		/// Show in a window
		imshow("Thres", thres_output);
		namedWindow("Contours", CV_WINDOW_AUTOSIZE);
		imshow("Contours", contour_output);
	}
}


void skeleton(Mat image) {
	Mat img = image.clone();
	cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);

	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp(img.size(), CV_8UC1);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		cv::morphologyEx(img, temp, cv::MORPH_OPEN, element);
		cv::bitwise_not(temp, temp);
		cv::bitwise_and(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		cv::erode(img, img, element);
		//cv::dilate(img, img, element);

		double max;
		cv::minMaxLoc(img, 0, &max);
		done = (max == 0);
	} while (!done);

	cv::imshow("Skeleton", skel);
	Mat dst;
	addWeighted(image,0.5, skel, 0.5, 0.0, dst);
	imshow("Combi", dst);
}
