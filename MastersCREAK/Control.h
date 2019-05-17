#pragma once

#define FUNCTION_ computeCREAK 
#define DO_EKF true
#define FAST_THRES_FREAK 8000
#define FAST_THRES_CREAK 8000
#define MATCH_THRES_ODOM 0.5			//Matching threshold for odometry
#define MATCH_THRES_ASSO 0.55			//Matching threshold for assosiation
#define MAX_FRAMES 1000					//amount of frames to process
#define MAX_DISTANCE 30					//Maximum distance a keypoint can be
#define MIN_TIMES_SEEN 4				//times that a landmark has to be seen
#define TRANSLATION_UNCERTAINTY 0.08	//how much we trust new landmarks
#define ROTATION_UNCERTAINTY 0.03		//how much we trust new landmarks
#define PARAM_PREDICT 0.5
#define PARAM_UPDATE 2
#define PARAM_NOISE 0.001

//Global Variables
Point3f currentPosition;
Point3f previousPosition;
Point3f previousPosition2;
Point3f currentRotation;
Point3f previousRotation;
Point3f previousRotation2;
Point3f currentVector;
