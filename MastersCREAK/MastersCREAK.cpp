#include "pch.h"
#include <stdio.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <bitset>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <thread>
#include <Windows.h>

#include "DescriptorTest.h"
#include "Odometry.h"
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



int main(int argc, char* argv[])
{
	///Odometry / SLAM---------------------------------------------------------------
	clock_t begin = clock();

	getOdometry();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	//int count = 0;
	//for (int i = 0; i < allLandmarks.size(); i++)
	//{
	//	if (allLandmarks.at(i).seen >= MIN_TIMES_SEEN)
	//		count++;
	//}
	//cout << "Landmarks: " << count << endl;

	cout << "True position: (" << xpositionTrue << "," << ypositionTrue << "," << zpositionTrue << ")" << endl;
	cout << "Odometry position: (" << currentPosition.x << "," << currentPosition.y << "," << currentPosition.z << ")" << endl;
	cout << endl;
	cout << "True rotation: (" << angleTrueY << "," << angleTrueP << ")" << endl;
	//cout << "Current rotation: (" << currentRotation.y << "," << currentRotation.x << ")" << endl;
	cout << "Odometry rotation: (" << angleOdomY << "," << angleOdomP << ")" << endl;

	cout << "Total time taken: " << elapsed_secs << "s\n\n" << endl;
	
	///CREAK / FREAK test--------------------------------------------------------------------------------
	/*clock_t begin = clock();
	testDescriptors("5");
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	begin = clock();
	testDescriptors("6");
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	/*begin = clock();
	testDescriptors("3");
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	begin = clock();
	testDescriptors("4");
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	begin = clock();
	testDescriptors("5");
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	begin = clock();
	testDescriptors("6");
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;*/

	cv::waitKey(0);
	return 0;
}
