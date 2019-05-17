#pragma once
#include "pch.h"
#include <stdio.h>
#include <iostream>
#include <thread>
#include <Windows.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "FREAK.h"
#include "CREAK.h"
#include "Control.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
#define MATCH_THRES_ 0.74
vector<DMatch> getBestMatches(Mat descriptors_Ref, Mat descriptors_Scene)
{
	BFMatcher matcher(cv::NORM_HAMMING);
	vector<vector<DMatch>> matches;
	vector<DMatch> best_Matches;
	matcher.knnMatch(descriptors_Ref, descriptors_Scene, matches, 2);

	//Retrieve the best matches
	for (size_t i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() < 2) continue;

		const DMatch &m1 = matches[i][0];
		const DMatch &m2 = matches[i][1];

		if (m1.distance <= MATCH_THRES_ * m2.distance) //Then it is a real match
			best_Matches.push_back(m1);
	}
	
	printf("Descriptor Size: %d\n", descriptors_Scene.rows);
	printf("Number of matches: %d\n", best_Matches.size());
	return best_Matches;
}

void showMatches(Mat img_1, Mat img_2, vector<KeyPoint> keypoints_Ref, vector<KeyPoint> keypoints_Scene, vector<DMatch> matches, String number)
{
	if (!img_1.empty() && !img_2.empty())
	{
		Mat img_matches;
		if (!keypoints_Ref.empty() && !keypoints_Scene.empty())
		{
			drawMatches(img_1, keypoints_Ref, img_2, keypoints_Scene,
				matches, img_matches, Scalar(230,230,0), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			imshow("Descriptor Matches " + number, img_matches);	
		}
	}
}

void testDescriptors(string number)
{
	Mat img_1, img_2;
	Mat descriptor_1, descriptor_2;
	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> best_Matches;

	///descriptors ----------------------------------------------------------------------------------
	///ORB
	Ptr<ORB> detector = ORB::create(1000000);
	///FAST
	//Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(20);
	//detector->setThreshold(FAST_THRES_FREAK);
	///----------------------------------------------------------------------------------------------
	String img_2_str = "Oxford\\Graffiti\\img" + number + ".ppm";
	img_1 = imread("Oxford\\Graffiti\\img1.ppm", IMREAD_COLOR);
	img_2 = imread(img_2_str, IMREAD_COLOR);

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	///Descriptors-----------------------------------------------------------------------------------
	///OpenCV FREAK
	//Ptr<FREAK> OpenCVFREAK = FREAK::create();
	//OpenCVFREAK->compute(img_1, keypoints_1, descriptor_1);
	//OpenCVFREAK->compute(img_2, keypoints_2, descriptor_2);
	///FREAK
	//computeFREAK(img_1, keypoints_1, descriptor_1);
	//computeFREAK(img_2, keypoints_2, descriptor_2);
	///CREAK
	computeCREAK(img_1, keypoints_1, descriptor_1);
	computeCREAK(img_2, keypoints_2, descriptor_2);
	///----------------------------------------------------------------------------------------------

	best_Matches = getBestMatches(descriptor_1, descriptor_2);
	showMatches(img_1, img_2, keypoints_1, keypoints_2, best_Matches, number);
}