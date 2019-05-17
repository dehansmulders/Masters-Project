#pragma once

#include "Landmark.h"
#include "Control.h"
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
int Psize = 6;
MatrixXd X; //Final State
MatrixXd Y; //landmark positions
MatrixXd Z; //Predicted landmark positions
MatrixXd Xu; //update state x, y, z, yaw, pitch, roll
MatrixXd Xu_; //previous update state x, y, z, yaw, pitch, roll
MatrixXd P;  //state covariance matrix
MatrixXd Q;  //Process Noise
MatrixXd Qi; //Measurement Noise
MatrixXd U; //Motion Noise
MatrixXd R; //Observation Noise
MatrixXd S; //Kalman innovation
MatrixXd Si;
MatrixXd K; //Kalman gain
MatrixXd dfv_dxv; //Jacobian dfv/dxv
MatrixXd F;
MatrixXd Fj;
MatrixXd G;
MatrixXd H;
MatrixXd df_dx;
MatrixXd df_du;
MatrixXd dh_dx;
MatrixXd R_x;

//J
void EKF_Jacobian_df_dx()
{
	/*double sy, sp, sr, cy, cp, cr;
	double xu, yu, zu;

	//get position increments
	xu = currentPosition.x - previousPosition.x;
	yu = currentPosition.y - previousPosition.y;
	zu = currentPosition.z - previousPosition.z;

	sy = sin(currentRotation.y);
	cy = cos(currentRotation.y);
	sp = sin(currentRotation.x);
	cp = cos(currentRotation.x);
	sr = sin(currentRotation.z);
	cr = cos(currentRotation.z);

	//get position increments
	xu = currentPosition.x - previousPosition.x;
	yu = currentPosition.y - previousPosition.y;
	zu = currentPosition.z - previousPosition.z;

	sy = sin(currentRotation.y);
	cy = cos(currentRotation.y);

	df_dx(0, 4) = -sy * sp*xu + sy * cp*sr*yu + sy * cp*cr*zu;								//x
	df_dx(2, 4) = -cy * sp*xu + cy * cp*sr*yu + cy * cp*cr*zu;								//z*/

	Xu_(0, 0) = previousPosition.x;
	Xu_(1, 0) = previousPosition.y;
	Xu_(2, 0) = previousPosition.z;

	Xu_(3, 0) = previousRotation.x;
	Xu_(4, 0) = previousRotation.y;
	Xu_(5, 0) = previousRotation.z;
	//dfv/dxv

	df_dx(0, 4) = Xu(0, 0)*sin((Xu(4, 0) - Xu_(4, 0))) / (1 + sin((Xu(4, 0) - Xu_(4, 0))));
	df_dx(2, 4) = Xu(2, 0)*sin((Xu(4, 0) - Xu_(4, 0))) / (1 + sin((Xu(4, 0) - Xu_(4, 0))));

	df_dx(4, 0) = (Xu(2, 0) - Xu_(2, 0)) / ((Xu(0, 0) - Xu_(0, 0))*(Xu(0, 0) - Xu_(0, 0)) + (Xu(2, 0) - Xu_(2, 0))*(Xu(2, 0) - Xu_(2, 0)));
	df_dx(4, 2) = -1 / (Xu_(2, 0) - Xu(2, 0) + (Xu_(2, 0)*(Xu(0, 0) - Xu_(0, 0))*(Xu(0, 0) - Xu_(0, 0)) - Xu(0, 0)*(Xu(0, 0) - Xu_(0, 0))*(Xu(0, 0) - Xu_(0, 0)) / 
												(Xu_(2, 0) - Xu(2, 0))*(Xu_(2, 0) - Xu(2, 0))));
	
	//df_dx(
	//x = x_ * cos(yaw_ - yaw)
	//z = z_ * sin(yaw_ - yaw)
	//yaw = atan((z - z_) / (x - x_))


	/*double sy, sp, sr, cy, cp, cr;
	double xu, yu, zu;

	//get position increments
	xu = currentPosition.x - previousPosition.x;
	yu = currentPosition.y - previousPosition.y;
	zu = currentPosition.z - previousPosition.z;

	sy = sin(currentRotation.y);
	cy = cos(currentRotation.y);
	sp = sin(currentRotation.x);
	cp = cos(currentRotation.x);
	sr = sin(currentRotation.z);
	cr = cos(currentRotation.z);

	df_dx(0, 3) = -sy * cp*xu + (-sy * sp*sr - cy * cr)*yu + (-sy * sp*cr + cy * sr)*zu;
	df_dx(0, 4) = -cy * sp*xu + cy * cp*sr*yu + cy * cp*cr*zu;								//z
	df_dx(0, 5) = (cy*sp*cr + sy * sr)*yu + (-cy * sp*sr + sy * cr)*zu;
	df_dx(1, 3) = cy * cp*xu + (cy*sp*sr - sy * cr)*yu + (cy*sp*cr + sy * sr)*zu;
	df_dx(1, 4) = -sy * sp*xu + sy * cp*sr*yu + sy * cp*cr*zu;								//x
	df_dx(1, 5) = (sy*sp*cr - cy * sr)*yu + (-sy * sp*sr - cy * cr)*zu;
	df_dx(2, 4) = -cp * xu - sp * sr*yu - sp * cr*zu;
	df_dx(2, 5) = cp * cr*yu - cp * sr*zu;*/
}

void EKF_Jacobian_df_du()
{
	//df_du = MatrixXd::Identity(6, 6)*200;

	U = MatrixXd::Identity(6, 6);
	double au, bu, cu;
	au = currentRotation.y;
	bu = currentRotation.x;
	cu = currentRotation.z;

	df_du(0, 0) = cos(au)*cos(bu);
	df_du(0, 1) = cos(au)*sin(bu)*sin(cu) - sin(au)*cos(cu);
	df_du(0, 2) = cos(au)*sin(bu)*cos(cu) + sin(au)*sin(cu);

	df_du(1, 0) = sin(au)*cos(bu);
	df_du(1, 1) = sin(au)*sin(bu)*sin(cu) + cos(au)*cos(cu);
	df_du(1, 3) = sin(au)*sin(bu)*cos(cu) - cos(au)*sin(cu);

	//df_du(2, 0) = -sin(bu);
	//df_du(2, 1) = cos(bu)*sin(cu);
	//df_du(2, 2) = cos(bu)*cos(cu);

	/*R_x.block(0, 0, 3, 3) = dfv_du;
	R_x(0, 3) = Xu(0, 0);
	R_x(1, 3) = Xu(1, 0);
	R_x(2, 3) = Xu(2, 0);
	R_x(3, 3) = 1;*/
}
//Jh
void EKF_Jacobian_dh_dx(int j, int i)
{
	double xt, yt, zt, yawt;
	double xr, yr, zr;
	double xw, yw, zw;
	xt = Xu(0, 0);
	yt = Xu(1, 0);
	zt - Xu(2, 0);
	yawt = Xu(4, 0);
	xr = observed.at(i).coord.x; //cos(yawt)*(xr - xt) - sin(yawt)*(zr - zt) + xt;
	yr = observed.at(i).coord.y;
	zr = observed.at(i).coord.z; //sin(yawt)*(xr - xt) + cos(yawt)*(zr - zt) + zt;

	if (yi.size() > 0)
	{
		xw = yi.at(j).coord.x;
		yw = yi.at(j).coord.y;
		zw = yi.at(j).coord.z;
	}
	else
	{
		xw = 0;
		yw = 0;
		zw = 0;
	}
	dh_dx.block(0, 0, 3, 3) = MatrixXd::Identity(3, 3);
	dh_dx.block(0, 6, 3, 3) = MatrixXd::Identity(3, 3);
	dh_dx(0, 4) = -xr * sin(yawt) + xt * sin(yawt) - zr * cos(yawt) + zt * cos(yawt);
	dh_dx(2, 4) = xr * cos(yawt) - xt * cos(yawt) - zr * sin(yawt) + zt * sin(yawt);

	/*MatrixXd dh_dx = MatrixXd::Zero(6, 3);

	//dhi_dyi
	for (int i = 0; i < observed.size(); i++)
	{
		double r = observed.at(i).distance;
		double x, y, z;
		x = observed.at(i).coord.x;
		y = observed.at(i).coord.y;
		z = observed.at(i).coord.z;

		dh_dx(0, 0) = x + Xu(0, 0);
		dh_dx(0, 1) = 0;
		dh_dx(0, 2) = 0;

		dh_dx(1, 0) = 0;
		dh_dx(1, 1) = y + Xu(1, 0);
		dh_dx(1, 2) = 0;

		dh_dx(2, 0) = 0;
		dh_dx(2, 1) = 0;
		dh_dx(2, 2) = z + Xu(2, 0);

		dh_dx(3, 0) = x * cos(Xu(3, 0));
		dh_dx(3, 1) = y + r * sin(Xu(3, 0));
		dh_dx(3, 2) = z * cos(Xu(3, 0));

		dh_dx(4, 0) = x + r * cos(Xu(4, 0));
		dh_dx(4, 1) = 0;
		dh_dx(4, 2) = z + r * sin(Xu(4, 0));

		dh_dx(5, 0) = 0;
		dh_dx(5, 1) = 0;
		dh_dx(5, 2) = 0;

	}*/
}

void initEKF()
{
	int n;
	if (FUNCTION_ == computeFREAK)
	{
		if (MIN_TIMES_SEEN == 2)
			n = 10104;
		if (MIN_TIMES_SEEN == 3)
			n = 934;
		if (MIN_TIMES_SEEN == 4)
			n = 700;// 481;// 747;
	}
	else if (FUNCTION_ == computeCREAK)
	{
		if (MIN_TIMES_SEEN == 2)
			n = 6517;
		if (MIN_TIMES_SEEN == 3)
			n = 1393;
		if (MIN_TIMES_SEEN == 4)
			n = 450;// 298;// 411;
	}

	Xu = MatrixXd::Zero(6, 1);
	Xu_ = MatrixXd::Zero(6, 1);
	P = MatrixXd::Zero(6 + 3 * n, 6 + 3 * n);
	X = MatrixXd::Zero(6 + 3 * n, 1);
	Y = MatrixXd::Zero(3 * n, 1);
	Z = MatrixXd::Zero(3 * n, 1);
	df_dx = MatrixXd::Identity(6, 6)*PARAM_PREDICT;
	df_du = MatrixXd::Identity(6 + 3 * n, 6 + 3 * n);
	dh_dx = MatrixXd::Zero(3, 9);
	F = MatrixXd::Zero(12, 6 + 3 * n);
	F.block(0, 0, 6, 6) = MatrixXd::Identity(6, 6);
	G = MatrixXd::Identity(6 + 3 * n, 6 + 3 * n);
	Q = MatrixXd::Identity(6 + 3 * n, 6 + 3 * n)*PARAM_NOISE;

	R = MatrixXd::Identity(3, 3);// *INITIAL_UNCERTAINTY;
	S = MatrixXd::Zero(6, 6);
	R_x = MatrixXd::Zero(4, 4);

}

void EKF_SetState()
{
	//populate Xu

	Xu(0, 0) = ((previousPosition.x + (previousPosition.x - previousPosition2.x)) + currentPosition.x) / 2;
	Xu(1, 0) = ((previousPosition.y + (previousPosition.y - previousPosition2.y)) + currentPosition.y) / 2;
	Xu(2, 0) = ((previousPosition.z + (previousPosition.z - previousPosition2.z)) + currentPosition.z) / 2;

	if (FUNCTION_ == computeCREAK)
	{
		Xu(3, 0) = (6 * (previousRotation.x + (previousRotation.x - previousRotation2.x)) + 7 * currentRotation.x) / 13;
		Xu(4, 0) = (4 * (previousRotation.y + (previousRotation.y - previousRotation2.y)) + 5 * currentRotation.y) / 9;
		Xu(5, 0) = (4 * (previousRotation.z + (previousRotation.z - previousRotation2.z)) + 5 * currentRotation.z) / 9;
	}
	else
	{
		Xu(3, 0) = (6 * (previousRotation.x + (previousRotation.x - previousRotation2.x)) + 7 * currentRotation.x) / 13;
		Xu(4, 0) = (6 * (previousRotation.y + (previousRotation.y - previousRotation2.y)) + 7 * currentRotation.y) / 13;
		Xu(5, 0) = (6 * (previousRotation.z + (previousRotation.z - previousRotation2.z)) + 7 * currentRotation.z) / 13;
	}
	//populate Y and Z;
	for (int i = 0; i < yi.size(); i++)
	{
		Y(3 * i, 0) = yi.at(i).coord.x;
		Y((3 * i) + 1, 0) = yi.at(i).coord.y;
		Y((3 * i) + 2, 0) = yi.at(i).coord.z;

		Z(3 * i, 0) = X(6 + 3 * i, 0);
		Z((3 * i) + 1, 0) = X(6 + (3 * i) + 1, 0);
		Z((3 * i) + 2, 0) = X(6 + (3 * i) + 2, 0);
	}

	//create X from Xu and Y
    X.block(0, 0, 6, 1) = Xu;
	X.block(6, 0, yi.size(), 1) = Y;

	//create covariance matrix U
	/*MatrixXd centered = Xu.transpose().colwise() - Xu.transpose().rowwise().mean();
	U = (centered.adjoint() * centered) / double(Xu.rows() - 1);*/
}

void EKF_Predict()
{
	if (!observed.empty())
	{
		//Update state
		EKF_SetState();

		//MatrixXd Pprev = P.block(0, 0, Psize, Psize);
		Psize = 6 + 3 * yi.size();
		//Update covariance
		EKF_Jacobian_df_dx(); //Create dfv/dx
		EKF_Jacobian_df_du();
		//G.block(0, 0, 6, 6 + 3 * yi.size()) = df_dx;// *F.block(0, 0, 12, 6 + 3 * yi.size());
		G.block(0, 0, 6, 6) = df_dx;
		//Q = df_du.block(0, 0, Psize, Psize) *U * df_du.block(0, 0, Psize, Psize).transpose();
	/*	MatrixXd GP = G.block(0, 0, Psize, Psize) * P.block(0, 0, Psize, Psize);
		MatrixXd GPGT = G.block(0, 0, Psize, Psize) * P.block(0, 0, Psize, Psize) * G.block(0, 0, Psize, Psize).transpose();
		MatrixXd GPGTpQ = (G.block(0, 0, Psize, Psize) * P.block(0, 0, Psize, Psize) * G.block(0, 0, Psize, Psize).transpose()) + Q.block(0, 0, Psize, Psize);*/
		P.block(0, 0, Psize, Psize) = G.block(0, 0, Psize, Psize) * P.block(0, 0, Psize, Psize) * G.block(0, 0, Psize, Psize).transpose() + Q.block(0, 0, Psize, Psize);

		//cout << Psize << endl;

		//cout << P.block(0, 0, Psize, Psize) << endl;
		if (P.hasNaN())//-NaN
		{
			//cout << "Previous" << endl;
			//cout << Pprev << endl;
			cout << "NaN" << endl;
			//cout << P.block(0, 0, Psize, Psize) << endl;
	
			//cout << "GP" << endl;
			//cout << GP << endl;
			//cout << "GPGT" << endl;
			//cout << GPGT << endl;
			//cout << "GPGTpQ" << endl;
			//cout << GPGTpQ << endl;
			//cout << G.block(0, 0, Psize, Psize) << endl;
			//cout << "G.transpose()" << endl;
			//cout << G.block(0, 0, 6 + 3 * yi.size(), 6 + 3 * yi.size()).transpose() << endl;
			//cout << "Q" << endl;
			//cout << Q << endl;

			system("pause");
		}

	}
}

void EKF_Update()
{
	//kalman innovation
	int j;
	if (!observed.empty())
	{
		for (int i = 0; i < observed.size(); i++)
		{
			if (existing.at(i).id == -1)
			{
				j = i; //location of corresponding landmark in X
				Fj = MatrixXd::Zero(9, 6 + 3 * observed.size());
				Fj.block(0, 0, 6, 6) = MatrixXd::Identity(6, 6);
				Fj.block(6, 3 * j + 3, 3, 3) = MatrixXd::Identity(3, 3);
			}
			else
			{
				j = allLandmarks.at(existing.at(i).id).id; //location of corresponding landmark in X
				Fj = MatrixXd::Zero(9, 6 + 3 * yi.size());
				Fj.block(0, 0, 6, 6) = MatrixXd::Identity(6, 6);
				Fj.block(6, 3 * j + 3, 3, 3) = MatrixXd::Identity(3, 3);
			}

			MatrixXd V = MatrixXd::Identity(3, 3);
			MatrixXd W = MatrixXd::Identity(3, 3) * PARAM_UPDATE;
			W(0, 0) = observed.at(i).distance;
			Qi = V * W * V.transpose();

			EKF_Jacobian_dh_dx(j, i);
			H = dh_dx * Fj;
			K = P.block(0, 0, Psize, Psize) * H.transpose() * (H*P.block(0, 0, Psize, Psize)*H.transpose() + Qi).inverse();
			//cout << Qi << endl;
			X.block(0, 0, Psize, 1) = X.block(0, 0, Psize, 1) + K*(Y.block(0, 0, 6 + Psize, 1) - Z.block(0, 0, Psize, 1));
			P.block(0, 0, Psize, Psize) = P.block(0, 0, Psize, Psize) - K*H*P.block(0, 0, Psize, Psize);
			
			/*cout << "X" << endl;
			cout << X.block(0, 0, 6, 1) << endl;
			cout << "Xu" << endl;
			cout << Xu << endl;*/
			//cout << Psize << endl;
		}

		currentPosition.x = X(0, 0);
		currentPosition.y = X(1, 0);
		currentPosition.z = X(2, 0);

		currentRotation.x = X(3, 0);
		currentRotation.y = X(4, 0);
		currentRotation.z = X(5, 0);

		for (int i = 0; i < yi.size(); i++)
		{
			allLandmarks.at(yi.at(i).id).coord.x = X(6 + 3 * i, 0);
			yi.at(i).coord.x = X(6 + 3 * i, 0);
			allLandmarks.at(yi.at(i).id).coord.y = X(6 + 3 * i + 1, 0);
			yi.at(i).coord.y = X(6 + 3 * i + 1, 0);
			allLandmarks.at(yi.at(i).id).coord.z = X(6 + 3 * i + 2, 0);
			yi.at(i).coord.z = X(6 + 3 * i + 2, 0);
		}
		
	}
	else
	{
		//cout << "No observed landmarks" << endl;
	}
}

