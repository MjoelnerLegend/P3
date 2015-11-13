#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\video\background_segm.hpp"
#include "Windows.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace cv;
using std::cin;
using std::cout;
using std::vector;
using std::endl;

//Functions.
void trackbarContent(int, void*);
void trackbar();
void morphologicalOpening(Mat &img);
void morphologicalClosing(Mat &img);
void blurTreshold(Mat &img);
void keyBindings(int key);
void showImgContours(Mat &threshImg, Mat &source);
void convexDefects(vector<Vec4i> convexityDefectsSet, vector<Point> contour, Mat &original);
void tracking(Mat &img, Mat &source);

//Range values:
int minHue = 0;
int maxhHue = 255;
int minSaturation = 0;
int maxSaturation = 255;
int minValue = 0;
int maxValue = 255;

//Object tracking variables.
int lastX = -1;
int lastY = -1;

//Treshold, dilate, erode and blur values.
int sizeErode = 1;
int sizeDilate = 1;
int sizeBlur = 1;
int thresholdValue = 0;

//Key bindings.
bool morph = false;
bool blurr = false;
bool switchView = false;
bool showContours = false;
bool showHull = false;
bool showCondefects = false;
bool trackObject = false;
bool keyLPressed = false;

int main(void) {
	trackbar();
	trackbarContent(0, 0);

	Mat frame, hsvFrame, rangeFrame, frameTmp;
	int key;
	VideoCapture cap(0);
	cap.read(frameTmp);
	Mat drawLine = Mat::zeros(frameTmp.size(), CV_8UC3);
	while ((key = waitKey(30)) != 27)
	{
		keyBindings(key);
		cap >> frame;
		flip(frame, frame, 180);
		cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

		inRange(hsvFrame, Scalar(minHue, minSaturation, minValue), Scalar(maxhHue, maxSaturation, maxValue), rangeFrame);

		if (morph) {
			morphologicalOpening(rangeFrame), morphologicalClosing(rangeFrame);
		}

		if (blurr) {
			blurTreshold(rangeFrame);
		}

		if (showContours) {
			showImgContours(rangeFrame, frame);
		}

		if (trackObject) {
			tracking(rangeFrame, frame);
		}

		if (switchView) {
			imshow("Camera", frame);
		}

		else {
			imshow("Camera", rangeFrame);
		}
	}
}

void trackbarContent(int, void*) {
	if (sizeErode == 0) {
		sizeErode = 1;
	}

	if (sizeDilate == 0) {
		sizeDilate = 1;
	}

	if (sizeBlur == 0) {
		sizeBlur = 1;
	}

	/*
	minHue = 80;
	maxhHue = 100;
	minSaturation = 60;
	maxSaturation = 255;
	minValue = 75;
	maxValue = 255;
	sizeDilate = 3;
	sizeErode = 3;
	*/
	
}

void trackbar() {
	String trackbarWindowName = "Trackbars";
	namedWindow(trackbarWindowName, CV_WINDOW_AUTOSIZE);
	createTrackbar("Hue min", trackbarWindowName, &minHue, maxhHue, trackbarContent);
	createTrackbar("Hue max", trackbarWindowName, &maxhHue, maxhHue, trackbarContent);
	createTrackbar("Saturation min", trackbarWindowName, &minSaturation, maxSaturation, trackbarContent);
	createTrackbar("Saturation max", trackbarWindowName, &maxSaturation, maxSaturation, trackbarContent);
	createTrackbar("Value min", trackbarWindowName, &minValue, maxValue, trackbarContent);
	createTrackbar("Value max", trackbarWindowName, &maxValue, maxValue, trackbarContent);
	createTrackbar("Erode", trackbarWindowName, &sizeErode, 31, trackbarContent);
	createTrackbar("Dilate", trackbarWindowName, &sizeDilate, 31, trackbarContent);
	createTrackbar("Blur", trackbarWindowName, &sizeBlur, 255, trackbarContent);
	createTrackbar("Threshold", trackbarWindowName, &thresholdValue, 255, trackbarContent);
}

void morphologicalOpening(Mat &img) {
	erode(img, img, getStructuringElement(MORPH_ELLIPSE, Size(sizeErode, sizeErode)));
	dilate(img, img, getStructuringElement(MORPH_ELLIPSE, Size(sizeDilate, sizeDilate)));
}

void morphologicalClosing(Mat &img) {
	dilate(img, img, getStructuringElement(MORPH_ELLIPSE, Size(sizeDilate, sizeDilate)));
	erode(img, img, getStructuringElement(MORPH_ELLIPSE, Size(sizeErode, sizeErode)));
}

void blurTreshold(Mat &img) {
	blur(img, img, Size(sizeBlur, sizeBlur), Point(-1, -1), BORDER_DEFAULT);
	threshold(img, img, thresholdValue, 255, THRESH_BINARY_INV);
}

void keyBindings(int key) {
	if (key == 'm') {
		morph = !morph;
	}

	if (key == 'b') {
		blurr = !blurr;
	}

	if (key == 'r') {
		switchView = !switchView;
	}

	if (key == 'c') {
		showContours = !showContours;
	}

	if (key == 'h') {
		showHull = !showHull;
	}

	if (key == 'd') {
		showCondefects = !showCondefects;
	}

	if (key == 't') {
		trackObject = !trackObject;
	}
}

void showImgContours(Mat &threshImg, Mat &source) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int largestArea = 0;
	int largestContourIndex = 0;

	findContours(threshImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	vector<vector<Point> >hull(contours.size());
	vector<vector<int> >inthull(contours.size());
	vector<vector<Vec4i> >defects(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hull[i], false);
		convexHull(Mat(contours[i]), inthull[i], false);
		if (inthull[i].size()>3) {
			convexityDefects(contours[i], inthull[i], defects[i]);
		}
	}

	for (int i = 0; i< contours.size(); i++) {
		double a = contourArea(contours[i], false);
		if (a>largestArea) {
			largestArea = a;
			largestContourIndex = i;
		}
	}

	if (contours.size() > 0) {
		drawContours(source, contours, largestContourIndex, CV_RGB(0, 255, 0), 2, 8, hierarchy);
		//drawContours(source, contours, -1, CV_RGB(0, 255, 0), 2, 8, hierarchy);
		if (showHull) {
			drawContours(source, hull, largestContourIndex, CV_RGB(0, 0, 255), 2, 8, hierarchy);
		}

		//drawContours(source, hull, -1, CV_RGB(0, 255, 0), 2, 8, hierarchy);
		if (showCondefects) {
			convexDefects(defects[largestContourIndex], contours[largestContourIndex], source);
		}
	}
}

void convexDefects(vector<Vec4i> convexityDefectsSet, vector<Point> contour, Mat &source)
{
	Point2f center;
	float radian;
	int fingers = 0;

	minEnclosingCircle(contour, center, radian);
	circle(source, center, 10, CV_RGB(0, 0, 255), 2, 8);
	for (int cDefIt = 0; cDefIt < convexityDefectsSet.size(); cDefIt++) {

		int startIndex = convexityDefectsSet[cDefIt].val[0]; Point ptStart(contour[startIndex]);

		int endIndex = convexityDefectsSet[cDefIt].val[1]; Point ptEnd(contour[endIndex]);

		int farIndex = convexityDefectsSet[cDefIt].val[2]; Point ptFar(contour[farIndex]);

		double depth = static_cast<double>(convexityDefectsSet[cDefIt].val[3]) / 256;

		line(source, ptStart, ptEnd, Scalar(0, 255, 255), 1);
		line(source, ptStart, ptFar, Scalar(0, 255, 255), 1);
		line(source, ptFar, ptEnd, Scalar(0, 255, 255), 1);

		if (depth>10 && ptStart.y<center.y) {
			circle(source, ptStart, 4, CV_RGB(255, 0, 0), 4);
			fingers++;
		}

		if (fingers > 5) {
			fingers = 5;
		}
	}
	putText(source, "Fingers : " + std::to_string(fingers), Point(50, 100), 2, 2, CV_RGB(30, 30, 30), 4, 8);
}

void tracking(Mat &img, Mat &source) {
	//Calculate the moments of the thresholded image
	Moments moment = moments(img);

	double drawM01 = moment.m01;
	double drawM10 = moment.m10;
	double drawArea = moment.m00;

	//If it's bigger than the stated there is no objects, otherwise there is so it runs the code below.
	if (drawArea > 10000) {
		//Calculate the position of the object and display.
		int posX = drawM10 / drawArea;
		int posY = drawM01 / drawArea;
		cout << "X = " << posX << endl;
		cout << "Y = " << posY << endl;
		if (posX > 175 && posX < 425){
			cout << "Center" << endl;
		}

		else if (posX < 175 && posY > 100 && posY < 400){
			cout << "Center left." << endl;
		}

		else if (posX > 425){
			cout << "Center right." << endl;
		}

		else if (posX < 175 && posY < 100){
			cout << "Top left." << endl;
		}

		else if (posX < 175 && posY > 400){
			cout << "Bottom left." << endl;
		}

		if (lastX >= 0 && lastY >= 0 && posX >= 0 && posY >= 0)
		{
			line(source, Point(posX, posY), Point(lastX, lastY), Scalar(255, 255, 255), 2);
		}

		lastX = posX;
		lastY = posY;
	}
}