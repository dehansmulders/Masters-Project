#pragma once
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

Mat ground = Mat::zeros(600, 600, CV_8UC3);
Mat odometry = Mat::zeros(600, 600, CV_8UC3); 
//Mat withSLAM = Mat::zeros(600, 600, CV_8UC3);
Mat orientation = Mat::zeros(600, 600, CV_8UC3);
Mat landMarks = Mat::zeros(600, 600, CV_8UC3);

double angleTrueY, angleTrueP, angleOdomY, angleOdomP;
double xpositionTrue, ypositionTrue, zpositionTrue;

struct coordinates
{
	double x;
	double y;
	double z;
};

struct landmark
{
	double distance;
	double yaw;
	double pitch;
	Point3f coord;
	Mat descriptor;
	int seen;//times seen
	float certainty;
	int id = -1;//used in assosiation
};

//Landmark Database
vector<landmark> newLandmarks;
vector<landmark> allLandmarks;

vector<landmark> yi; //landmarks to append to the EKF
//Landmark Database Descriptors Mat
Mat newLandmarksMat;
Mat allLandmarksMat;

vector<landmark> observed; //array of newly matched landmarks
vector<landmark> existing; //array of existing landmarks to which they were matched

struct Quat
{
	float w, x, y, z;
};

Point3f normalize(Point3f vector)
{
	float norm = sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
	vector.x /= norm;
	vector.y /= norm;
	vector.z /= norm;

	return vector;
}

Quat normalizeQuaternion(Quat q)
{
	float norm = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
	q.w /= norm;
	q.x /= norm;
	q.y /= norm;
	q.z /= norm;

	return q;
}

Quat conjugateQuaternion(Quat q)
{
	q.x = -q.x;
	q.y = -q.y;
	q.z = -q.z;

	return q;
}

Quat multiplyQuaternions(Quat q1, Quat q2)
{
	Quat ans;

	/*ans.x = q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
	ans.y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
	ans.z = q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
	ans.w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;*/

	ans.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
	ans.x = q1.x*q2.w + q1.w*q2.x + q1.y*q2.z - q1.z*q2.y;
	ans.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
	ans.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;

	return ans;
}

float length(Point3f v)
{
	return(sqrtf(v.x*v.x + v.y*v.y + v.z*v.z));
}

float length2d(Vec2d v)
{
	return(sqrtf(v[0]*v[0] + v[1]*v[1]));
}

float dotProduct(Point3f v1, Point3f v2)
{
	return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

Point3f crossProduct(Point3f v1, Point3f v2)
{
	return Point3f((v1.y*v2.z - v1.z*v2.y) , -(v1.x*v2.z - v1.z*v2.x) , (v1.x*v2.y - v1.y*v2.x)); //not sure about -j
}

Quat getRotation(Point3f vector1, Point3f vector2)
{
	Quat q, qNorm;
	Point3f xyz = crossProduct(vector1, vector2);
	q.x = xyz.x;
	q.y = xyz.y;
	q.z = xyz.z;
	//q.w = sqrtf(length(vector1)*length(vector1)*length(vector2)*length(vector2) + dotProduct(vector1, vector2));
	q.w = sqrtf(length(vector1)*length(vector1)*length(vector2)*length(vector2)) + dotProduct(vector1, vector2);

	qNorm = normalizeQuaternion(q);

	return qNorm;
}

//insert yaw, pitch, roll as (x, y, z)
Quat eulerToQuat(double yaw, double pitch, double roll)
{
	double cy = cos(yaw * 0.5);
	double sy = sin(yaw * 0.5);
	double cp = cos(pitch * 0.5);
	double sp = sin(pitch * 0.5);
	double cr = cos(roll * 0.5);
	double sr = sin(roll * 0.5);

	Quat q;
	q.w = cy * cp * cr + sy * sp * sr;
	q.x = cy * cp * sr - sy * sp * cr;
	q.y = sy * cp * sr + cy * sp * cr;
	q.z = sy * cp * cr - cy * sp * sr;
	return q;
}

//returns yaw, pitch, roll (x, y, z)
Point3f quatToEuler(Quat q)
{
	Point3f euler;
	// roll (x-axis rotation)
	double sinr_cosp = +2.0 * (q.w * q.x + q.y * q.z);
	double cosr_cosp = +1.0 - 2.0 * (q.x * q.x + q.y * q.y);
	euler.z = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	double sinp = +2.0 * (q.w * q.y - q.z * q.x);
	if (fabs(sinp) >= 1)
		euler.y = copysign(CV_PI / 2, sinp); // use 90 degrees if out of range
	else
		euler.y = asin(sinp);

	// yaw (z-axis rotation)
	double siny_cosp = +2.0 * (q.w * q.z + q.x * q.y);
	double cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z);
	euler.x = atan2(siny_cosp, cosy_cosp);

	return euler;
}


Point3f rotateVector(Point3f vector, Quat q)
{
	q = normalizeQuaternion(q);
	Quat qv, p;
	Quat vec;
	vec.w = 0;
	vec.x = vector.x;
	vec.y = vector.y;
	vec.z = vector.z;

	qv = multiplyQuaternions(q, vec); 
	p = multiplyQuaternions(qv, conjugateQuaternion(q)); //p = q*.v.q

	Point3f ret;
	ret.x = p.x;
	ret.y = p.y;
	ret.z = p.z;

	return ret;
}

Mat eulerToR(Point3f euler)
{
	Mat R;
	double au, bu, cu;
	au = euler.z;
	bu = euler.y;
	cu = euler.x;
	double R11, R12, R13, R21, R22, R23, R31, R32, R33;
	R11 = cos(bu)*cos(au);
	R12 = cos(au)*sin(bu)*sin(cu) - sin(au)*cos(cu);
	R13 = cos(au)*sin(bu)*cos(cu) + sin(au)*sin(cu);
	R21 = sin(au)*cos(bu);
	R22 = sin(au)*sin(bu)*sin(cu) + cos(au)*cos(cu);
	R23 = sin(au)*sin(bu)*cos(cu) - cos(au)*sin(cu);
	R31 = -sin(bu);
	R32 = cos(bu)*sin(cu);
	R33 = cos(bu)*cos(cu);


	R = Mat_<double>(3, 3) << R11, R12, R23,
							R21, R22, R23,
							R31, R32, R33;

	return R;
}

//Point3f RtoEuler(Mat R, int Option = 0)
//Option parameter: 0 = 1st, 1 = 2nd
Point3f RtoEuler(Mat R, int option = 0)
{
	Point3f euler;
	double a, b, c;
	if (abs(R.at<double>(3, 1)) != 1)
	{
		if (option == 0)
		{
			b = -asin(R.at<double>(3, 1));

			c = atan2((R.at<double>(3, 2) / cos(b)), (R.at<double>(3, 3) / cos(b)));

			a = atan2((R.at<double>(2, 1) / cos(b)), (R.at<double>(1, 1) / cos(b)));
		}
		else if (option == 1)
		{
			b = CV_PI + asin(R.at<double>(3, 1));

			c = atan2((R.at<double>(3, 2) / cos(b)), (R.at<double>(3, 3) / cos(b)));

			a = atan2((R.at<double>(2, 1) / cos(b)), (R.at<double>(1, 1) / cos(b)));
		}
	}
	else
	{
		if (R.at<double>(3, 1) == -1)
		{
			b = CV_PI / 2.0;
			c = atan2(R.at<double>(1, 2), R.at<double>(1, 3));
			a = 0;
		}
		else
		{
			b = -CV_PI / 2.0;
			c = atan2(-R.at<double>(1, 2), -R.at<double>(1, 3));
			a = 0;
		}
	}
	euler.x = c;
	euler.y = b;
	euler.z = a;
	return euler;
}

void appendLandmark(vector<Mat> descriptors, vector<Point2f> points_t0, vector<Point2f> points_t1, 
	Mat oldT, Mat finalT, Point3f prevPrevPosition, Point3f prevPosition, Point3f currPosition, 
	double scale, Mat oldOrientation, Mat newOrientation)
{	//points_t1 belong to descriptors, points_t0 is previous points
	//relativeT shows translation from last iteration t0 to t1, R is rotation
	//finalT is total translation thus far
	//image resolution 1241 x 376
	double focal = 718.8560; //focal length in pixels
	Point2d pp(607.1928, 185.2157); //center-point of image
	Point3f referenceVec(0.0, 0.0, 1.0); //straight on z-axis, z represents forward straight.

	newLandmarks.clear();
	newLandmarksMat.release();//empties the variable

	float deltaX1 = prevPosition.x - prevPrevPosition.x;
	float deltaY1 = prevPosition.y - prevPrevPosition.y;
	float deltaZ1 = prevPosition.z - prevPrevPosition.z;

	float deltaX2 = currPosition.x - prevPosition.x;
	float deltaY2 = currPosition.y - prevPosition.y;
	float deltaZ2 = currPosition.z - prevPosition.z;

	Point3f vectorP1(deltaX1, deltaY1, deltaZ1);//Direction facing at t-1
	Point3f vectorP2(deltaX2, deltaY2, deltaZ2);//Direction facing at t
	Point3f P1 = prevPosition;
	Point3f P2 = currPosition;

	Point3f vectorOld(oldOrientation.at<double>(0), oldOrientation.at<double>(1), oldOrientation.at<double>(2));
	Point3f vectorNew(newOrientation.at<double>(0), newOrientation.at<double>(1), newOrientation.at<double>(2));

	//Quat reference_u = getRotation(referenceVec, vectorP2);
	Quat rotation_u = getRotation(referenceVec, vectorOld);//vectorP1 also works
	Quat rotation_v = getRotation(referenceVec, vectorNew);//vectorP2 also works

	//currentVector = vectorNew;
	//currentRotation = quatToEuler(rotation_v);
	//previousRotation = quatToEuler(rotation_u);
	Mat R = eulerToR(currentRotation);

	//printf("currentRotation: (%2f, %2f, %2f)\n\n", R.at<double>(0), R.at<double>(1), R.at<double>(2));
	//printf("rot: (%2f, %2f, %2f, %2f)\n\n", rotation_u.w, rotation_u.x, rotation_u.y, rotation_u.z);

	//displays current orientation
	if (false)
	{
		orientation = Mat::zeros(600, 600, CV_8UC3);
		Point3f vec(0.0, 0.0, 1.0);
		Point3f lineVec = rotateVector(vec, rotation_v);
		Point one(300, 300);
		//Point two(300 + 200 * cos(currentRotation.x), 300 - 200 * sin(currentRotation.x)); //pitch
		Point two(300 + 200 * sin(currentRotation.y), 300 - 200 * cos(currentRotation.y)); //yaw
		//Point two(300 + 200 * cos(currentRotation.z), 300 - 200 * sin(currentRotation.z)); //roll
		line(orientation, one, two, Scalar(255, 0, 255), 5);
		imshow("test", orientation);
	}
	Point3f w(deltaX2, deltaY2, deltaZ2);
	//landMarks = Mat::zeros(600, 600, CV_8UC3);
	for (int i = 0; i < descriptors.size(); i++)
	{
		//position time = t; (u)
		double dx1 = points_t0.at(i).x - pp.x;
		double dy1 = pp.y - points_t0.at(i).y;
		//double dx1 = pp.x - points_t0.at(i).x;
		//double dy1 = points_t0.at(i).y - pp.y;
		double dz1 = focal;

		Point3f relative_u(dx1, dy1, dz1);
		Point3f u = rotateVector(relative_u, rotation_u);

			//next position time = t+1; (v)
		double dx2 = points_t1.at(i).x - pp.x;
		double dy2 = pp.y - points_t1.at(i).y;
		//double dx2 = pp.x - points_t1.at(i).x;
		//double dy2 = points_t1.at(i).y - pp.y;
		double dz2 = focal;

		Point3f relative_v(dx2, dy2, dz2);
		Point3f v = rotateVector(relative_v, rotation_v);

		double a = dotProduct(u, u);
		double b = dotProduct(u, v);
		double c = dotProduct(v, v);
		double d = dotProduct(u, w);
		double e = dotProduct(v, w);

		double s = (b*e - c*d) / (a*c - b*b);
		double t = (a*e - b*d) / (a*c - b*b);

		Point3f landMark_s((P1.x + s*u.x), (P1.y + s*u.y), (P1.z + s*u.z));
		Point3f landMark_t((P2.x + t*v.x), (P2.y + t*v.y), (P2.z + t*v.z));

		Point3f directionVec((P1.x - landMark_s.x), (P1.y - landMark_s.y), (P1.z - landMark_s.z)); //Vector from position to landmark
		double distance = sqrt(directionVec.x*directionVec.x + directionVec.y*directionVec.y + directionVec.z*directionVec.z);
		//double yaw = atan(directionVec.y / directionVec.x);
		double pitch = -atan(directionVec.z / sqrt(directionVec.x*directionVec.x + directionVec.y*directionVec.y));
		if (distance <= MAX_DISTANCE) //we dont want landmarks that are too far
		{
			//printf("S Landmark Coordinate at: P(%2f , %2f , %2f)\n", landMark_s.x, landMark_s.y, landMark_s.z);
			//printf("T Landmark Coordinate at: P(%2f , %2f , %2f)\n\n", landMark_t.x, landMark_t.y, landMark_t.z);
			double dot = dotProduct(directionVec, vectorP2);
			/*if (dot > 0)//then behind
			{
				landMark_s.x = P1.x + directionVec.x;//flip point about position
				//landMark_s.y = P1.y + directionVec.y;
				landMark_s.z = P1.z + directionVec.z;
			}*/
			if (dot < 0)//then not behind, if behind something is not right
			{
				landmark newLandmark;
				newLandmark.distance = distance;
				//newLandmark.yaw = yaw;
				//newLandmark.pitch = pitch;
				newLandmark.coord = landMark_s;
				newLandmark.descriptor = descriptors.at(i).row(0);
				newLandmark.seen = 1;
				//newLandmark.certainty = INITIAL_UNCERTAINTY;

				newLandmarks.push_back(newLandmark);
				//printf("appending mat\n");
				newLandmarksMat.push_back(descriptors.at(i).row(0));

				//Draw Landmarks
				int posx = int(finalT.at<double>(0));
				int posy = int(finalT.at<double>(2));

				int x = int(landMark_s.x);
				int y = int(landMark_s.z);
				//circle(landMarks, Point(300 + x, 500 - y), 1, Scalar(255, 0, 0), 1);
				circle(landMarks, Point(300 + posx, 500 - posy), 1, Scalar(255, 255, 0), 1);
				imshow("Land Marks", landMarks);
			}
		}
	}
}


