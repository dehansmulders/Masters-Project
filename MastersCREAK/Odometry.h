#pragma once
#include "FREAK.h"
#include "CREAK.h"
#include "Landmark.h"
#include "SLAM.h"
#include "Control.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

double trueX, trueY;
double getAbsoluteScale(int frame_id, int sequence_id)
{
	string line;
	int i = 0;
	ifstream myfile("KITTI\\00\\00.txt");
	double x = 0, y = 0, z = 0, r = 0;
	double x_prev, y_prev, z_prev;
	double scale;
	if (myfile.is_open())
	{
		while ((getline(myfile, line)) && (i <= frame_id))
		{
			z_prev = z;
			x_prev = x;
			y_prev = y;
			istringstream in(line);
			for (int j = 0; j < 12; j++)
			{
				in >> z;
				if (j == 7) y = z;
				if (j == 3)  x = z;
				if (j == 11)  r = z;
			}
			i++;
			trueX = x;
			trueY = z;
			scale = sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));
		}
		myfile.close();
	}
	else
	{
		cout << "Unable to open file for some reason..";
		return 0;
	}
	circle(odometry, Point(300 + x, 500 - z), 1, Scalar(0, 0, 255), 1);
	//circle(odometry, Point(50 + x, 550 - z), 1, Scalar(0, 0, 255), 2);

	xpositionTrue = x;
	ypositionTrue = y;
	zpositionTrue = z;
	Vec2d v1(0, 1);
	Vec2d v2(x - x_prev, z - z_prev);
	Vec2d v3(1, 0);
	Vec2d v4(z - z_prev, y - y_prev);
	angleTrueY = acos(v1.dot(v2) / (length2d(v1)*length2d(v2)));
	angleTrueP = acos(v3.dot(v4) / (length2d(v3)*length2d(v4)));

	//printf("True Coordinates: x = %02fm z = %02fm height = %02fm\n", x, z, y);
	return scale;
}

//Matches(query, train)
double avgLandmarks = 0;
vector<DMatch> getMatches(Mat descriptors_1, Mat descriptors_2)//Matches Features
{
	//printf("query size: %d\n", descriptors_1.rows);
	//printf("train size: %d\n", descriptors_2.rows);
	//clock_t begin = clock();

	BFMatcher matcher(cv::NORM_HAMMING);
	vector<vector<DMatch>> matches;
	vector<DMatch> best_matches, best_matches_2, bOb;
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

	//Threshold matches
	for (size_t i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() < 2)
			continue;

		const DMatch &m1 = matches[i][0];
		const DMatch &m2 = matches[i][1];

		if (m1.distance <= MATCH_THRES_ODOM * m2.distance) //Then it is a good match
			best_matches.push_back(m1);
	}
	/*
	//repeat backwards
	matcher.knnMatch(descriptors_2, descriptors_1, matches, 2);
	for (size_t i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() < 2)
			continue;

		const DMatch &m1 = matches[i][0];
		const DMatch &m2 = matches[i][1];

		if (m1.distance <= 0.6 * m2.distance) //Then it is a good match
			best_matches_2.push_back(m1);
	}

	if (best_matches.size == best_matches_2.size)
	{
		for (int i = 0; i < best_matches.size; i++)
		{
			if (best_matches.at(i). == best_matches_2.at(i))
			{
				bOb.push_back(best_matches);
			}
		}
	}
	else
		printf("Woops, not the same size");
		*/
		//clock_t end = clock();
		//double elapsed_secs = double(end - begin) * 1000 / CLOCKS_PER_SEC;
		//cout << "Total time taken: " << elapsed_secs << "ms" << endl;
		//printf("Matches: %d\n\n", best_matches.size());
	//printf("matches size: %d\n", best_matches.size());
	avgLandmarks += (best_matches.size());
	return best_matches;
}

void featureTracking(Mat descriptor_1, Mat descriptor_2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<Point2f>& points_1, vector<Point2f>& points_2, vector<Mat>& bestDesc)
{
	//Match descriptors on second image
	vector<DMatch> matches = getMatches(descriptor_2, descriptor_1);
	//only keep keypoints for good matches
	vector<KeyPoint> best_keypoints_1, best_keypoints_2;
	for (int i = 0; i < matches.size(); i++)
	{
		int q = matches.at(i).queryIdx; //current image
		int t = matches.at(i).trainIdx; //previous image

		best_keypoints_1.push_back(keypoints_1.at(t));
		best_keypoints_2.push_back(keypoints_2.at(q));
		//printf("mat rows: %d\n", descriptor_1.rows);
		bestDesc.push_back(descriptor_2.row(q));
	}

	KeyPoint::convert(best_keypoints_1, points_1, vector<int>());
	KeyPoint::convert(best_keypoints_2, points_2, vector<int>());
}


void featureDetection(Mat img_1, vector<KeyPoint>& keypoints_1, Mat& descriptor_1)
{
	//Ptr<ORB> detector = ORB::create(5000); //use FAST or ORB?
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(20); //use FAST or ORB? fast is faster and more accurate.
	if (FUNCTION_ == computeFREAK)
		detector->setThreshold(FAST_THRES_FREAK);
	else
		detector->setThreshold(FAST_THRES_CREAK);
	detector->detect(img_1, keypoints_1);
	//printf("Keypoints: %d\n", keypoints_1.size());
	FUNCTION_(img_1, keypoints_1, descriptor_1);
}

double errorX = 0;
double errorY = 0;

void getOdometry()
{
	namedWindow("View", WINDOW_AUTOSIZE);
	namedWindow("SLAM", WINDOW_AUTOSIZE);
	//namedWindow("Ground Truth", WINDOW_AUTOSIZE);
	odometry.setTo(Scalar(230, 230, 230));
	//legend
	circle(odometry, Point(440, 25), 6 , Scalar(200, 255, 0), FILLED);
	putText(odometry, "FREAK EKF-SLAM", Point(450, 30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 0), 1.0);
	circle(odometry, Point(440, 46), 6, Scalar(0, 0, 255), FILLED);
	putText(odometry, "Ground Truth", Point(450, 51), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 0), 1.0);

	//KITTI dataset paramaters
	double focal = 718.8560;
	Point2d pp(607.1928, 185.2157);

	Point3f prevPrevPosition, prevPosition, currPosition;
	Mat finalRot, finalTrans, relativeTrans;
	ofstream outputFile;
	outputFile.open("Output.txt");
	double scale = 1.00;
	char imgName_1[200];
	char imgName_2[200];
	sprintf_s(imgName_1, "KITTI\\00\\image_2\\%06d.png", 0);
	sprintf_s(imgName_2, "KITTI\\00\\image_2\\%06d.png", 1);

	//Read images from dataset
	Mat img_1 = imread(imgName_1);
	Mat img_2 = imread(imgName_2);

	if (!img_1.data || !img_2.data) {
		std::cout << "Error reading images" << std::endl; 
		if (!img_1.data)
			std::cout << "img1" << std::endl;
		if (!img_2.data)
			std::cout << "img2" << std::endl;
		return;
	}

	//Get features
	vector<Point2f> points_1, points_2;
	vector<Mat> bestDescriptors;
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat desc_1, desc_2;
	featureDetection(img_1, keypoints_1, desc_1);
	featureDetection(img_2, keypoints_2, desc_2);

	//Get location of points_1 on img_1 and points_2 in img_2
	featureTracking(desc_1, desc_2, keypoints_1, keypoints_2, points_1, points_2, bestDescriptors);

	//Get essential matrix and recover pose estimation
	Mat E, R, t, mask;
	E = findEssentialMat(points_2, points_1, focal, pp, RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points_2, points_1, R, t, focal, pp, mask);

	//Mat prevImage = img_2.clone();
	Mat currImage;
	Mat prevFeatures = desc_2.clone();
	Mat currFeatures;
	Mat oldTrans, newTrans, oldOrientation, newOrientation, newRot, oldRot;

	vector<Point2f> prevPoints;
	vector<Point2f> currPoints;
	vector<KeyPoint> prevKeypoints = keypoints_2;
	vector<KeyPoint> currKeypoints;
	char filename[100];

	finalRot = R.clone();
	finalTrans = t.clone();
	oldTrans = finalTrans;
	newOrientation = finalRot * t;
	newTrans = scale * newOrientation;

	currPosition = Point3f(newTrans.at<double>(0), newTrans.at<double>(1), newTrans.at<double>(2));
	prevPosition = currPosition;
	prevPrevPosition = prevPosition;
	currentPosition = Point3f(0.0, 0.0, 0.0);
	currentVector = Point3f(0.0, 0.0, 1.0);
	Mat prevTrans;
	double distance = 0;
	for (int frame = 0; frame < MAX_FRAMES; frame++)
	{
		bestDescriptors.clear();
		sprintf_s(filename, "KITTI\\00\\image_2\\%06d.png", frame);
		currImage = imread(filename);

		//featureDetection(prevImage, prevKeypoints);
		featureDetection(currImage, currKeypoints, currFeatures);
		featureTracking(prevFeatures, currFeatures, prevKeypoints, currKeypoints, prevPoints, currPoints, bestDescriptors);
		//printf("Keypoints: %d\n", prevPoints.size());

		if (prevPoints.size() >= 8)//Essential matrix needs minimum of 8
		{
			E = findEssentialMat(currPoints, prevPoints, focal, pp, RANSAC, 0.999, 1.0, mask);
			recoverPose(E, currPoints, prevPoints, R, t, focal, pp, mask);
			//printf("size: %d\n", R.rows); //R and t have 3 variables

			scale = getAbsoluteScale(frame, 0);
			distance += scale;
			//printf("Scale: %2f\n", scale);
			//printf("t: %dx%d\n", t.rows, t.cols);
			if ((scale > 0.1))// No need to concern if distance travelled is too small
			{
				/*Mat difference = scale * (finalRot * t);
				prevTrans = finalTrans;
				finalTrans = finalTrans + difference;
				finalRot = R * finalRot;

				currentPosition = Point3f(finalTrans.at<double>(0), finalTrans.at<double>(1), finalTrans.at<double>(2));*/
				oldTrans = finalTrans;
				oldOrientation = newOrientation;
				newOrientation = finalRot * t;	
				finalTrans = finalTrans + scale * newOrientation;
				oldRot = finalRot;
				finalRot = R * finalRot;
				newRot = finalRot - oldRot;

				prevPrevPosition = prevPosition;
				prevPosition = currPosition;
				currPosition = Point3f(finalTrans.at<double>(0), finalTrans.at<double>(1), finalTrans.at<double>(2));

				previousPosition2 = previousPosition;
				previousPosition = currentPosition;
				currentPosition = currPosition;

				Mat euler;
				Rodrigues(finalRot, euler);
				previousRotation2 = previousRotation;
				previousRotation = currentRotation;
				currentRotation = Point3f(euler.at<double>(0), euler.at<double>(1), euler.at<double>(2));// p, y, r
				//printf("Rotation: (%2f, %2f, %2f)\n", currentRotation.x, currentRotation.y, currentRotation.z);
				//printf("w: %2f, %2f, %2f, %2f\n", getRotation(prevPrevPosition, prevPosition).w, getRotation(prevPrevPosition, prevPosition).x, getRotation(prevPrevPosition, prevPosition).y, getRotation(prevPrevPosition, prevPosition).z);
				//append descriptor to database
				
				if (DO_EKF)
				{
					if (frame <= 950)
					{
						appendLandmark(bestDescriptors, prevPoints, currPoints, oldTrans, finalTrans, prevPrevPosition, prevPosition, currPosition, scale, oldOrientation, newOrientation);
						computeSLAM(finalRot, finalTrans);//perform SLAM to alter the currently thought location
					}
				}

				finalTrans.at<double>(0) = currentPosition.x;
				finalTrans.at<double>(1) = currentPosition.y;
				finalTrans.at<double>(2) = currentPosition.z;

				euler.at<double>(0) = currentRotation.x;
				euler.at<double>(1) = currentRotation.y;
				euler.at<double>(2) = currentRotation.z;

				Rodrigues(euler, finalRot);
			}

			//Draw Odometry
			//int x = int(finalTrans.at<double>(0));
			//int y = int(finalTrans.at<double>(2));

			int x = int(currentPosition.x);
			int y = int(currentPosition.z);
			Vec2d v1(0, 1);
			Vec2d v2(currentPosition.x - previousPosition.x, currentPosition.z - previousPosition.z);
			Vec2d v3(1, 0);
			Vec2d v4(currentPosition.z - previousPosition.z, currentPosition.y - previousPosition.y);
			angleOdomY = acos(v1.dot(v2) / (length2d(v1)*length2d(v2)));
			angleOdomP = acos(v3.dot(v4) / (length2d(v3)*length2d(v4)));

			//printf("Error: %2fx, %2fy\n", errorX, errorY);
			//printf("Total E: %2fx, %2fy\n", (trueX - x), (trueY - y));

			//finalTrans.at<double>(0) = trueX;
			//finalTrans.at<double>(2) = trueY;

			circle(odometry, Point(300 + x, 500 - y), 1, Scalar(200, 255, 0), 1);
			//circle(odometry, Point(50 + x, 550 - y), 1, Scalar(255, 150, 0), 2);
			for (int i = 0; i < currPoints.size(); i++)
			{
				circle(currImage, Point2f(currPoints.at(i).x, currPoints.at(i).y), 3, Scalar(0, 0, 255), 1);
				line(currImage, Point2f(currPoints.at(i).x, currPoints.at(i).y), Point2f(prevPoints.at(i).x, prevPoints.at(i).y), Scalar(255, 255, 0), 1);
			}

		}
		prevFeatures = currFeatures.clone();
		prevKeypoints = currKeypoints;

		imshow("View", currImage);
		imshow("SLAM", odometry);

		//cout << frame << ": " << "True position: (" << xpositionTrue << "," << ypositionTrue << "," << zpositionTrue << ")" << endl;
		//cout << frame << ": " << "Odom position: (" << currentPosition.x << "," << currentPosition.y << "," << currentPosition.z << ")" << endl;
		//cout << "Total Distance: " << distance << endl;
		cout << "Average Landmarks: " << avgLandmarks << endl;
		//cout << "Odometry rotation: (" << angleOdomY << "," << angleOdomP << ")" << endl;

		//printf("Coordinates: x = %02fm y = %02fm z = %02fm\n", finalTrans.at<double>(0), finalTrans.at<double>(2), finalTrans.at<double>(1));
		//outputFile << finalTrans.at<double>(0) << " " << finalTrans.at<double>(1) << " " << finalTrans.at<double>(2) << endl;

		if (cv::waitKey(1) == 27)
			return;
	}
}

