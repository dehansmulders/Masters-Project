#pragma once
#include "Landmark.h"
#include "Control.h"
#include "EKF.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include <Eigen/Dense>
#include <Eigen/LU>

using Eigen::VectorXd;
using Eigen::MatrixXd;
bool init = true;

void makeAssosiations(Mat newDescriptors, Mat &descriptors)//Matches landmarks
{
	BFMatcher matcher(cv::NORM_HAMMING);
	vector<vector<DMatch>> matches;
	matcher.knnMatch(newDescriptors, descriptors, matches, 2);
	//printf("Matches: %d\n", matches.size());
	//Threshold matches
	for (size_t i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() < 2)
			continue;

		const DMatch &m1 = matches[i][0];
		const DMatch &m2 = matches[i][1];

		int q = m1.queryIdx; //new descriptors id
		int t = m1.trainIdx; //existing descriptors id
		if (m1.distance <= MATCH_THRES_ASSO * m2.distance) //Then it is a good match
		{
			//printf("matched: %2f, %2f, %2f\n", allLandmarks.at(t).coord.x, allLandmarks.at(t).coord.y, allLandmarks.at(t).coord.z);
			allLandmarks.at(t).seen++;
			landmark temp = allLandmarks.at(t);
			
			temp.id = t;
			int seen = temp.seen;
			if (seen >= MIN_TIMES_SEEN)
			{
				observed.push_back(newLandmarks.at(q));
				existing.push_back(temp);

				int x = int(newLandmarks.at(q).coord.x);
				int y = int(newLandmarks.at(q).coord.z);
				circle(landMarks, Point(300 + x, 500 - y), 1, Scalar(100, 0, 255), 1);

				imshow("Land Marks", landMarks);
			}
			if (seen == MIN_TIMES_SEEN)
			{
				allLandmarks.at(t).id = yi.size();
				yi.push_back(temp);//new landmarks that passed seen threshold added to y		
			}
		}
		else
		{
			//printf("Not matched\n");
			allLandmarks.push_back(newLandmarks.at(q));
			descriptors.push_back(newDescriptors.row(q));
		}
	}


}

void EKF()
{
	EKF_Predict();
	EKF_Update();
}

//My way!
void SLAM()
{
	double x_r = currentPosition.x;
	double y_r = currentPosition.y;
	double z_r = currentPosition.z;

	double x_l_o, y_l_o, z_l_o;
	double x_l_e, y_l_e, z_l_e;
	double yaw_o, range_o;
	double yaw_e, range_e;
	double range_u;
	int seen;

	double u_x, u_y, u_z;
	double u_pitch, u_yaw;

	double d_x = 0;
	double d_y = 0;
	double d_z = 0;
	double d_yaw = 0;

	double d_range = 0;
	for (int i = 0; i < observed.size(); i++)
	{
		//for observed landmarks (z_l_o - z_r)
		x_l_o = observed.at(i).coord.x;
		y_l_o = observed.at(i).coord.y;
		z_l_o = observed.at(i).coord.z;
		yaw_o = atan((z_l_o - z_r) / (x_l_o - x_r));

		//for existing landmarks
		seen = existing.at(i).seen;
		x_l_e = existing.at(i).coord.x;
		y_l_e = existing.at(i).coord.y;
		z_l_e = existing.at(i).coord.z;
		yaw_e = atan((z_l_e - z_r) / (x_l_e - x_r));

		//Updated values
		u_x = (x_l_o + x_l_e * seen) / (seen + 1);
		u_y = (y_l_o + y_l_e * seen) / (seen + 1);
		u_z = (z_l_o + z_l_e * seen) / (seen + 1);
		u_yaw = (yaw_o + yaw_e * seen) / (seen + 1);

		//range
		range_e = sqrt((z_l_e - z_r)*(z_l_e - z_r) + (y_l_e - y_r)*(y_l_e - y_r) + (x_l_e - x_r)*(x_l_e - x_r));
		range_u = sqrt((u_z - z_r)*(u_z - z_r) + (u_y - y_r)*(u_y - y_r) + (u_x - x_r)*(u_x - x_r));

		d_range += (range_u / range_e) / observed.size();

		//Difference
		d_x += (u_x - x_l_e) / observed.size();
		d_y += (u_y - y_l_e) / observed.size();
		d_z += (u_z - z_l_e) / observed.size();
		d_yaw += (u_yaw - yaw_e) / observed.size();

		//write new Landmark coords
		int index = existing.at(i).id;
		if (index == -1)
			index = i;

		allLandmarks.at(index).coord.x = u_x;
		allLandmarks.at(index).coord.y = u_y;
		allLandmarks.at(index).coord.z = u_z;
	}
	//Apply Robot changes

	//currentPosition.x += d_x * TRANSLATION_UNCERTAINTY;
	//currentPosition.y += d_y * TRANSLATION_UNCERTAINTY;
	//currentPosition.z += d_z * TRANSLATION_UNCERTAINTY;
	
	//currentRotation.x += d_pitch * ROTATION_UNCERTAINTY;
	//currentRotation.y -= d_yaw * ROTATION_UNCERTAINTY;
	
	//moving average method
//current observed change
	double dx, dy, dz, dyaw;
	dx = previousPosition.x - previousPosition2.x;
	dy = previousPosition.y - previousPosition2.y;
	dz = previousPosition.z - previousPosition2.z;
	dyaw = previousRotation.y - previousRotation2.y;

	//predicted odometry
	double px, py, pz, pyaw;
	px = previousPosition.x + dx;
	py = previousPosition.y + dy;
	pz = previousPosition.z + dz;
	pyaw = previousRotation.y + dyaw;

	//adjust odometry;
	/*currentPosition.x = (currentPosition.x + px + (currentPosition.x + d_x * TRANSLATION_UNCERTAINTY)) / 3.0;
	currentPosition.y = (currentPosition.y + py + (currentPosition.y + d_y * TRANSLATION_UNCERTAINTY)) / 3.0;
	currentPosition.z = (currentPosition.z + pz + (currentPosition.z + d_z * TRANSLATION_UNCERTAINTY)) / 3.0;
	currentRotation.y = (currentRotation.y + pyaw + (currentRotation.y + d_yaw * ROTATION_UNCERTAINTY)) / 3.0;*/

	currentPosition.x = d_range * TRANSLATION_UNCERTAINTY * currentPosition.x;
	currentPosition.y = d_range * TRANSLATION_UNCERTAINTY * currentPosition.y;
	currentPosition.z = d_range * TRANSLATION_UNCERTAINTY * currentPosition.z;
	currentRotation.y = (currentRotation.y + pyaw + (currentRotation.y + d_yaw * ROTATION_UNCERTAINTY)) / 3.0;
}

//Mat orientation = Mat::zeros(600, 600, CV_8UC3);
void computeSLAM(Mat &finalRot, Mat&finalTrans)
{
	if (init)
	{
		initEKF();
		init = false;
		return;
	}

	if (allLandmarks.empty()) //then Populate all new landmarks into database
	{
		if (!newLandmarks.empty())
		{
			//printf("landmarks size before: %d\n", allLandmarksMat.rows);
			allLandmarksMat = newLandmarksMat;
			allLandmarks = newLandmarks;
			observed = newLandmarks;
			existing = allLandmarks;

			EKF();
			//SLAM();
		}
	}
	else
	{
		observed.clear();
		existing.clear();
		makeAssosiations(newLandmarksMat, allLandmarksMat);//observed is new matched landmarks, existing are those they matched to		

		EKF();
		//SLAM();
	}	
}
