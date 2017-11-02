/*
CS585_lab8_aquarium.cpp
Aythors: Prateek Mehta, Siva Perumal, Shruti Kannan

CS585 Image and Video Computing Fall 2017
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

using namespace cv;
using namespace std;



struct tracked_obj {
	Point2d last;
	Point2d current;
	Point2d projected;
	bool tracked;
	Scalar color;
};

struct centroid {
	Point2d coords;
	bool isTracked;
};

void file_processing(Mat& frame_curr, vector <vector<int>>& data_store, Mat& binary3C, vector<centroid>& centroids);
void alpha_beta(tracked_obj *pos);
void greedyTrack(vector<tracked_obj>& pos, vector<centroid>& centroids);
void drawTrajectory(vector<tracked_obj>& pos, Mat &canvas);


double MINIMUM_TRACKING_RADIUS = 20;

double a = .85; //1.5
double b = .005; //0.5
double v_x;
double v_x_1 = 0;
double projected_x_last = 0;
double projected_x;
double v_y;
double v_y_1 = 0;
double projected_y_last = 0;
double projected_y;

/**
TODO (or TAKEHOME): Read the segmented images of the aquarium dataset and create the following:
a) A binary image where background pixels are labelled 0, and object pixels are
labelled 255
b) A labelled image where background pixels are labelled 0, and object pixels are
labelled numerically
c) A 3-channel image where the object centroids are drawn as red dots on the binary image
*/
int main(int argc, char **argv)
{
	//Mat frame_prev = imread("Segmentation_Aqua/2014_aq_segmented 01.jpg");
	VideoWriter video("aqua.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(449, 450));

	int initial = 01;
	int final = 53;
	int iter = final - (initial)+1;
	vector <vector <int>> file_data, file_data1, file_data2;
	vector<centroid> centroids;
	vector<tracked_obj> track_obj;
	Scalar Colors[] = { Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,0,255) };
	Mat binary_try;
	binary_try = Mat::zeros(Size(449, 450), CV_8UC3);

	for (int i = 0; i < iter; i++) {
		centroids.clear();
		string file1;
		string ini = "Segmentation_Aqua/2014_aq_segmented ";
		//cout << "OK" << endl;

		string fin = ".jpg";
		string num1 = static_cast<ostringstream*>(&(ostringstream() << (initial + i)))->str();

		if (num1.size() == 1) num1 = "0"+num1;
		else if (num1.size() == 2) num1 = num1;

		//cout << num1 << " " << endl;
		file1.append(ini); file1.append(num1); file1.append(fin);

		cout << file1 << " " << endl;

		Mat frame_curr = imread(file1);

		Mat blob;
		blob = frame_curr.clone();
		//cout << frame_curr.rows = 450<< frame_curr.cols = 449 << endl;

		// create windows

		//namedWindow("Original");
		imshow("Original", frame_curr);

		Mat binary3C;

		//finding contours
		file_processing(frame_curr, file_data1, binary3C ,centroids);

		for (int k = 0; k<track_obj.size(); k++)
		{
			alpha_beta(&track_obj[k]);
		}
		greedyTrack(track_obj, centroids);


		int flag = 0;
		for (int j = 0; j < centroids.size(); j++) {
			//loop through untracked centroids
			if (!centroids.at(j).isTracked) {
				flag++;
				tracked_obj newObj;
				//naively set current, last, and projected all to be the centroid's points
				newObj.current = centroids.at(j).coords;
				newObj.last = centroids.at(j).coords;
				newObj.projected = centroids.at(j).coords;
				int blue = rand() % 254;
				int green = rand() % 254;
				int red = rand() % 254;
				newObj.color = Colors[j % 6]; //Colors[j%9];
											  //cout << "New color is (" << blue << "," << green << "," << red << ")" << endl;
				track_obj.push_back(newObj);
			}
		}

		//drawTrajectory(trackedBats,frame);
		drawTrajectory(track_obj, binary_try);
		cout << binary_try.rows << binary_try.cols << endl;
		cout << binary3C.rows << binary3C.cols << endl;
		add(binary3C, binary_try, binary3C);
		//vid.write(blob);
		imshow("Trajectory", binary3C);
		video.write(binary3C);
		//char keey = waitKey(0);
		waitKey(20);
		cout << "Number of new bats: " << flag << " in frame number " << i << endl;
		//assigning current frame values to the previous frame
		//frame_prev = frame_curr;
		//file_data = file_data1;
		//imshow("Mask2", binary3C);
		waitKey(20);
	}
	video.release();
	waitKey(0);

	//return 0;
}

void file_processing(Mat& frame_curr, vector <vector<int>>& data_store, Mat& binary3C, vector<centroid>& centroids) {

	Mat3b hsv;
	cvtColor(frame_curr, hsv, COLOR_BGR2HSV);

	Mat1b mask1, mask2;
	inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
	inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

	Mat1b mask = mask1 | mask2;

	//imshow("Mask", mask);
	
	cvtColor(mask, binary3C, CV_GRAY2BGR);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Get the moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(contours.size());
	vector <int> datarow;


	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		//cout << mc[i] << endl;
		circle(binary3C, mc[i], 3, Scalar(0, 0, 255), -1, 8);
		datarow.push_back(mc[i].x);
		datarow.push_back(mc[i].y);

		data_store.push_back(datarow);
		datarow.clear();
		//cout << file_data.size() << endl;
	}
	centroid cent;
	centroids.clear();
	for (int i = 0; i < data_store.size(); ++i)
	{
		Point2d center(data_store.at(i).at(0), data_store.at(i).at(1));
		cent.coords = center;
		cent.isTracked = false;
		centroids.push_back(cent);
	}
}

void alpha_beta(tracked_obj *pos) {
	// projected distance moved
	projected_x = projected_x_last + v_x_1;
	v_x = v_x_1;
	projected_y = projected_y_last + v_y_1;
	v_y = v_y_1;

	// Update the residuals
	double r_x = pos->current.x - projected_x;
	double r_y = pos->current.y - projected_y;

	// Update projections
	projected_x += a * r_x;
	v_x += (b * r_x);
	projected_y += a * r_y;
	v_y += (b * r_y);

	// Set new last projections to current
	projected_x_last = projected_x;
	v_x_1 = v_x;
	projected_y_last = projected_y;
	v_y_1 = v_y;


	// Update x and y projections
	pos->projected.x = projected_x;
	pos->projected.y = projected_y;
}

void greedyTrack(vector<tracked_obj>& pos, vector<centroid>& centroids) {
	int maxX = 450;
	int maxY = 449;

	for (int i = 0; i < pos.size(); i++) {
		int projX = pos.at(i).projected.x;
		int projY = pos.at(i).projected.y;
		double minDif = 200000; //arbitrary large number (bigger than maximum distance in 450x449)
		int centroidIndex = -1;
		//remove if the bat is projected outside the bounds.
		if (projX > maxX || projY > maxY || projX < 0 || projY < 0) {
			pos.erase(pos.begin() + i);
			//cout << "erased OOB bat" << endl;
		}
		else {
			for (int j = 0; j < centroids.size(); j++) {
				//loop through untracked centroids
				if (!centroids.at(j).isTracked) {
					double distance = sqrt(pow(abs(centroids.at(j).coords.x - projX), 2) + pow(abs(centroids.at(j).coords.y - projY), 2));
					//cout << "distance = " << distance << endl;
					if (distance < minDif) {
						//keep track of smallest difference
						minDif = distance;
						centroidIndex = j;
						//   cout << "centroid index set to " << centroidIndex << endl;
					}
				}
			}
			if (minDif > MINIMUM_TRACKING_RADIUS) {
				for (int j = 0; j < centroids.size(); j++) {
					//loop through tracked centroids
					if (centroids.at(j).isTracked) {
						double distance = sqrt(pow(abs(centroids.at(j).coords.x - projX), 2) + pow(abs(centroids.at(j).coords.y - projY), 2));
						if (distance < minDif) {
							//keep track of smallest difference
							minDif = distance;
							centroidIndex = j;
						}
					}
				}
			}
			//make sure we actually found a centroid (Note - centroidIndex defaults to -1)
			if (centroidIndex == -1) {
				//if we ran out of centroids, delete this tracked bat. Note super accurate, but the greedy alg works
				pos.erase(pos.begin() + i);
				cout << "centroid index = -1" << endl;
			}

			else {
				pos.at(i).last = pos.at(i).current;
				pos.at(i).current = centroids.at(centroidIndex).coords;
				centroids.at(centroidIndex).isTracked = true;
				
			}
		}
	}
}

void drawTrajectory(vector<tracked_obj>& pos, Mat &canvas)
{
	int thickness = 2;
	int lineType = 8;
	for (int i = 0; i<pos.size(); i++)
	{
		line(canvas, pos[i].last, pos[i].current, pos[i].color, thickness, lineType);
	}
}

