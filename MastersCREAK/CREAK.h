#pragma once
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
#include "CREAK.h"
#include "FREAK.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//mutex mux;

const int CREAKnbPairs = 192; //Size of descriptor, CREAK is 192 FREAK is 512
const int CREAKnbColour = CREAKnbPairs / 3; //Bits used by each colour channel
const int CREAKnbOrienpairs = 57;//CREAK has 57, FREAK has 45


DescriptionPair CREAKdescriptionPairs[CREAKnbColour];
DescriptionPair CREAKdescriptionPairsB[CREAKnbColour];
DescriptionPair CREAKdescriptionPairsG[CREAKnbColour];
DescriptionPair CREAKdescriptionPairsR[CREAKnbColour];
OrientationPair CREAKorientationPairs[CREAKnbOrienpairs];
vector<PatternPoint> CREAKpatternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and nbOrientation)

bool extAll; // true if all pairs need to be extracted for pairs selection

//default pairs
const int CREAKdefPairsB[CREAKnbColour] =
{
	549, 359, 114, 561, 536, 886, 487, 705,
	499, 574, 806, 360, 517, 113, 719, 578,
	147, 684, 355, 523, 52, 44, 250, 61,
	372, 424, 134, 192, 844, 796, 8, 466,
	214, 142, 212, 794, 26, 388, 271, 194,
	335, 689, 555, 46, 366, 354, 484, 172,
	230, 108, 385, 898, 312, 723, 136, 695,
	843, 206, 808, 89, 11, 404, 867, 678
};

const int CREAKdefPairsG[CREAKnbColour] =
{
	354, 214, 26, 312, 689, 543, 555, 561,
	108, 782, 574, 147, 724, 706, 366, 114,
	194, 424, 192, 134, 487, 52, 142, 867,
	844, 230, 511, 523, 723, 806, 784, 8,
	339, 359, 11, 250, 113, 212, 372, 800,
	872, 578, 61, 878, 136, 695, 388, 690,
	895, 843, 484, 341, 625, 360, 736, 271,
	801, 583, 684, 466, 773, 549, 50, 78
};

const int CREAKdefPairsR[CREAKnbColour] =
{
	843, 543, 895, 11, 470, 484, 52, 862,
	108, 806, 857, 55, 8, 782, 230, 339,
	625, 44, 724, 113, 583, 149, 736, 341,
	466, 214, 194, 883, 176, 418, 172, 548,
	271, 198, 312, 329, 730, 250, 327, 773,
	114, 236, 549, 487, 218, 684, 784, 78,
	706, 354, 723, 555, 388, 366, 232, 431,
	678, 689, 61, 566, 660, 838, 866, 580
};
const int CREAKdefPairs[CREAKnbPairs] =
{
	404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
	560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
	592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
	796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
	691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
	381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
	382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
	466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
	418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
	72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
	56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
	129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
	236,26,172,147,574,561,32,294,
	/*429,724,755,398,787,288,299,
	769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
	544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
	212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
	194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
	276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
	844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
	736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
	182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
	242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
	819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
	185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
	851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
	13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
	413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
	197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
	41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
	152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
	260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
	131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
	325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
	670,249,36,581,389,605,331,518,442,822*/
};

void buildCREAKPattern()
{
	if (patternScale == patternScale0 && nOctaves == nOctaves0 && !CREAKpatternLookup.empty())
		return;

	nOctaves0 = nOctaves;
	patternScale0 = patternScale;

	CREAKpatternLookup.resize(nbScales*nbOrientation*nbPoints);
	double scaleStep = pow(2.0, (double)(nOctaves) / nbScales); // 2 ^ ( (nOctaves-1) /nbScales)
	double scalingFactor, alpha, beta, theta = 0;

	// pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
	const int n[8] = { 6,6,6,6,6,6,6,1 }; // number of points on each concentric circle (from outer to inner)
	const double bigR(2.0 / 3.0); // bigger radius
	const double smallR(2.0 / 24.0); // smaller radius
	const double unitSpace((bigR - smallR) / 21.0); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)

	// radii of the concentric cirles (from outer to inner)
	const double radius[8] = { bigR,
								bigR - 6 * unitSpace,
								bigR - 11 * unitSpace,
								bigR - 15 * unitSpace,
								bigR - 18 * unitSpace,
								bigR - 20 * unitSpace,
								smallR,
								0.0 };

	// sigma of pattern points (each group of 6 points on a concentric cirle has the same sigma)
	//FREAK has each size at 2.0
	const double sigma[8] = { radius[0] / 2.0,
								radius[1] / 2.0,
								radius[2] / 2.0,
								radius[3] / 4.0,//4.0
								radius[4] / 2.0,
								radius[5] / 4.0,//4.0
								radius[6] / 4.0,//4.0
								radius[6] / 4.0 };//4.0

	// fill the lookup table
	for (int scaleIdx = 0; scaleIdx < nbScales; ++scaleIdx)
	{
		patternSizes[scaleIdx] = 0; // proper initialization
		scalingFactor = pow(scaleStep, scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

		for (int orientationIdx = 0; orientationIdx < nbOrientation; ++orientationIdx)
		{
			theta = double(orientationIdx) * 2 * CV_PI / double(nbOrientation); // nbOrientation of the pattern
			int pointIdx = 0;

			PatternPoint* patternLookupPtr = &CREAKpatternLookup[0];
			for (size_t i = 0; i < 8; ++i)
			{
				for (int k = 0; k < n[i]; ++k)
				{
					beta = CV_PI / n[i] * (i % 2); // nbOrientation offset so that groups of points on each circles are staggered
					alpha = double(k) * 2 * CV_PI / double(n[i]) + beta + theta;

					// add the point to the look-up table
					PatternPoint& point = patternLookupPtr[scaleIdx*nbOrientation*nbPoints + orientationIdx * nbPoints + pointIdx];
					point.x = static_cast<float>(radius[i] * cos(alpha) * scalingFactor * patternScale);
					point.y = static_cast<float>(radius[i] * sin(alpha) * scalingFactor * patternScale);
					point.sigma = static_cast<float>(sigma[i] * scalingFactor * patternScale);

					// adapt the sizeList if necessary
					const int sizeMax = static_cast<int>(ceil((radius[i] + sigma[i])*scalingFactor*patternScale)) + 1;
					if (patternSizes[scaleIdx] < sizeMax)
						patternSizes[scaleIdx] = sizeMax;

					++pointIdx;
				}
			}
		}
	}

	// build the list of nbOrientation pairs
	CREAKorientationPairs[0].i = 0; CREAKorientationPairs[0].j = 3; CREAKorientationPairs[1].i = 1; CREAKorientationPairs[1].j = 4; CREAKorientationPairs[2].i = 2; CREAKorientationPairs[2].j = 5;
	CREAKorientationPairs[3].i = 0; CREAKorientationPairs[3].j = 2; CREAKorientationPairs[4].i = 1; CREAKorientationPairs[4].j = 3; CREAKorientationPairs[5].i = 2; CREAKorientationPairs[5].j = 4;
	CREAKorientationPairs[6].i = 3; CREAKorientationPairs[6].j = 5; CREAKorientationPairs[7].i = 4; CREAKorientationPairs[7].j = 0; CREAKorientationPairs[8].i = 5; CREAKorientationPairs[8].j = 1;

	CREAKorientationPairs[9].i = 6; CREAKorientationPairs[9].j = 9; CREAKorientationPairs[10].i = 7; CREAKorientationPairs[10].j = 10; CREAKorientationPairs[11].i = 8; CREAKorientationPairs[11].j = 11;
	CREAKorientationPairs[12].i = 6; CREAKorientationPairs[12].j = 8; CREAKorientationPairs[13].i = 7; CREAKorientationPairs[13].j = 9; CREAKorientationPairs[14].i = 8; CREAKorientationPairs[14].j = 10;
	CREAKorientationPairs[15].i = 9; CREAKorientationPairs[15].j = 11; CREAKorientationPairs[16].i = 10; CREAKorientationPairs[16].j = 6; CREAKorientationPairs[17].i = 11; CREAKorientationPairs[17].j = 7;

	CREAKorientationPairs[18].i = 12; CREAKorientationPairs[18].j = 15; CREAKorientationPairs[19].i = 13; CREAKorientationPairs[19].j = 16; CREAKorientationPairs[20].i = 14; CREAKorientationPairs[20].j = 17;
	CREAKorientationPairs[21].i = 12; CREAKorientationPairs[21].j = 14; CREAKorientationPairs[22].i = 13; CREAKorientationPairs[22].j = 15; CREAKorientationPairs[23].i = 14; CREAKorientationPairs[23].j = 16;
	CREAKorientationPairs[24].i = 15; CREAKorientationPairs[24].j = 17; CREAKorientationPairs[25].i = 16; CREAKorientationPairs[25].j = 12; CREAKorientationPairs[26].i = 17; CREAKorientationPairs[26].j = 13;

	CREAKorientationPairs[27].i = 18; CREAKorientationPairs[27].j = 21; CREAKorientationPairs[28].i = 19; CREAKorientationPairs[28].j = 22; CREAKorientationPairs[29].i = 20; CREAKorientationPairs[29].j = 23;
	CREAKorientationPairs[30].i = 18; CREAKorientationPairs[30].j = 20; CREAKorientationPairs[31].i = 19; CREAKorientationPairs[31].j = 21; CREAKorientationPairs[32].i = 20; CREAKorientationPairs[32].j = 22;
	CREAKorientationPairs[33].i = 21; CREAKorientationPairs[33].j = 23; CREAKorientationPairs[34].i = 22; CREAKorientationPairs[34].j = 18; CREAKorientationPairs[35].i = 23; CREAKorientationPairs[35].j = 19;

	CREAKorientationPairs[36].i = 24; CREAKorientationPairs[36].j = 27; CREAKorientationPairs[37].i = 25; CREAKorientationPairs[37].j = 28; CREAKorientationPairs[38].i = 26; CREAKorientationPairs[38].j = 29;
	CREAKorientationPairs[39].i = 30; CREAKorientationPairs[39].j = 33; CREAKorientationPairs[40].i = 31; CREAKorientationPairs[40].j = 34; CREAKorientationPairs[41].i = 32; CREAKorientationPairs[41].j = 35;
	CREAKorientationPairs[42].i = 36; CREAKorientationPairs[42].j = 39; CREAKorientationPairs[43].i = 37; CREAKorientationPairs[43].j = 40; CREAKorientationPairs[44].i = 38; CREAKorientationPairs[44].j = 41;

	// Additional CREAK pairs
	CREAKorientationPairs[45].i = 0; CREAKorientationPairs[45].j = 6; CREAKorientationPairs[46].i = 1; CREAKorientationPairs[46].j = 7; CREAKorientationPairs[47].i = 2; CREAKorientationPairs[47].j = 8;
	CREAKorientationPairs[48].i = 3; CREAKorientationPairs[48].j = 9; CREAKorientationPairs[49].i = 4; CREAKorientationPairs[49].j = 10; CREAKorientationPairs[50].i = 5; CREAKorientationPairs[50].j = 11;

	CREAKorientationPairs[51].i = 6; CREAKorientationPairs[51].j = 13; CREAKorientationPairs[52].i = 7; CREAKorientationPairs[52].j = 14; CREAKorientationPairs[53].i = 8; CREAKorientationPairs[53].j = 15;
	CREAKorientationPairs[54].i = 9; CREAKorientationPairs[54].j = 16; CREAKorientationPairs[55].i = 10; CREAKorientationPairs[55].j = 17; CREAKorientationPairs[56].i = 11; CREAKorientationPairs[56].j = 12;

	for (unsigned m = CREAKnbOrienpairs; m--; )
	{
		const float dx = CREAKpatternLookup[CREAKorientationPairs[m].i].x - CREAKpatternLookup[CREAKorientationPairs[m].j].x;
		const float dy = CREAKpatternLookup[CREAKorientationPairs[m].i].y - CREAKpatternLookup[CREAKorientationPairs[m].j].y;
		const float norm_sq = (dx*dx + dy * dy);
		CREAKorientationPairs[m].weight_dx = int((dx / (norm_sq))*4096.0 + 0.5);
		CREAKorientationPairs[m].weight_dy = int((dy / (norm_sq))*4096.0 + 0.5);
	}

	// build the list of description pairs
	vector<DescriptionPair> allPairs;
	for (unsigned int i = 1; i < (unsigned int)nbPoints; ++i)
	{
		// (generate all the pairs)
		for (unsigned int j = 0; (unsigned int)j < i; ++j)
		{
			DescriptionPair pair = { (uchar)i,(uchar)j };
			allPairs.push_back(pair);
		}
	}

	for (int i = 0; i < CREAKnbPairs; ++i)
	{
		CREAKdescriptionPairs[i] = allPairs[CREAKdefPairs[i]];
		//CREAKdescriptionPairsB[i] = allPairs[CREAKdefPairsB[i]];
		//CREAKdescriptionPairsG[i] = allPairs[CREAKdefPairsG[i]];
		//CREAKdescriptionPairsR[i] = allPairs[CREAKdefPairsR[i]];
	}

}

template <typename srcMatType>
void extractCREAKDescriptor(srcMatType pointsValue[3][43], void ** ptr)
{
	bitset<CREAKnbPairs>** ptrScalar = (bitset<CREAKnbPairs>**) ptr;

	// extracting descriptor
	for (int n = 0; n < CREAKnbPairs - 1; n += 3) //default
	{
		(*ptrScalar)->set(n, pointsValue[0][CREAKdescriptionPairs[n].i] >= pointsValue[0][CREAKdescriptionPairs[n].j]);
		(*ptrScalar)->set(n + 1, pointsValue[1][CREAKdescriptionPairs[n + 1].i] >= pointsValue[1][CREAKdescriptionPairs[n + 1].j]);
		(*ptrScalar)->set(n + 2, pointsValue[2][CREAKdescriptionPairs[n + 2].i] >= pointsValue[2][CREAKdescriptionPairs[n + 2].j]);
	}
	/*int cnt = 0;
		for (int n = 0; n < CREAKnbPairs-1; n += 3)//learned pairs
		{
			(*ptrScalar)->set(n, pointsValue[0][CREAKdescriptionPairsB[cnt].i] >= pointsValue[0][CREAKdescriptionPairsB[cnt].j]);
			(*ptrScalar)->set(n+1, pointsValue[1][CREAKdescriptionPairsG[cnt].i] >= pointsValue[1][CREAKdescriptionPairsG[cnt].j]);
			(*ptrScalar)->set(n+2, pointsValue[2][CREAKdescriptionPairsR[cnt].i] >= pointsValue[2][CREAKdescriptionPairsR[cnt].j]);
			cnt++;
		}*/

	--(*ptrScalar);
}

template <typename imgType, typename iiType>
imgType meanCREAKIntensity(InputArray _image, InputArray _integral,
	const float kp_x,
	const float kp_y,
	const unsigned int scale,
	const unsigned int rot,
	const unsigned int point)
{
	Mat image = _image.getMat(), integral = _integral.getMat();
	// get point position in image
	const PatternPoint& FreakPoint = CREAKpatternLookup[scale*nbOrientation*nbPoints + rot * nbPoints + point];
	const float xf = FreakPoint.x + kp_x;
	const float yf = FreakPoint.y + kp_y;
	const int x = int(xf);
	const int y = int(yf);

	// get the sigma:
	const float radius = FreakPoint.sigma;

	// calculate output:
	if (radius < 0.5)
	{
		// interpolation multipliers:
		const int r_x = static_cast<int>((xf - x) * 1024);
		const int r_y = static_cast<int>((yf - y) * 1024);
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);
		unsigned int ret_val;
		// linear interpolation:
		ret_val = r_x_1 * r_y_1*int(image.at<imgType>(y, x))
			+ r_x * r_y_1*int(image.at<imgType>(y, x + 1))
			+ r_x_1 * r_y  *int(image.at<imgType>(y + 1, x))
			+ r_x * r_y  *int(image.at<imgType>(y + 1, x + 1));
		//return the rounded mean
		ret_val += 2 * 1024 * 1024;
		return static_cast<imgType>(ret_val / (4 * 1024 * 1024));
	}

	// calculate borders
	const int x_left = int(xf - radius + 0.5);
	const int y_top = int(yf - radius + 0.5);
	const int x_right = int(xf + radius + 1.5);//integral image is 1px wider
	const int y_bottom = int(yf + radius + 1.5);//integral image is 1px higher
	iiType ret_val;

	ret_val = integral.at<iiType>(y_bottom, x_right);//bottom right corner
	ret_val -= integral.at<iiType>(y_bottom, x_left);
	ret_val += integral.at<iiType>(y_top, x_left);
	ret_val -= integral.at<iiType>(y_top, x_right);
	const int area = (x_right - x_left) * (y_bottom - y_top);
	ret_val = (ret_val + area / 2) / area;
	//~ cout<<integral.step[1]<<endl;
	return static_cast<imgType>(ret_val);
}

template <typename srcMatType, typename iiMatType>
void computeCREAKDescriptors(InputArray _image, vector<KeyPoint>& keypoints, OutputArray _descriptors)
{
	Mat image = _image.getMat();
	Mat greyImage;
	cvtColor(image, greyImage, COLOR_BGR2GRAY);

	Mat splitImage[3];
	split(image, splitImage);
	Mat imgIntegral[3];//BGR

	integral(splitImage[0], imgIntegral[0], DataType<iiMatType>::type);
	integral(splitImage[1], imgIntegral[1], DataType<iiMatType>::type);
	integral(splitImage[2], imgIntegral[2], DataType<iiMatType>::type);

	//imshow("B", splitImage[0]);
	//imshow("G", splitImage[1]);
	//imshow("R", splitImage[2]);

	vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
	const vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in vector erase function
	const vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in vector erase function
	const float sizeCst = static_cast<float>(nbScales / (nbLog2* nOctaves));

	srcMatType pointsValue[3][nbPoints];

	int thetaIdx = 0;
	int direction0;
	int direction1;

	// compute the scale index corresponding to the keypoint size and remove keypoints close to the border
	if (scaleNormalized)
	{
		for (size_t k = keypoints.size(); k--; )
		{
			//Is k non-zero? If so, decrement it and continue"
			kpScaleIdx[k] = max((int)(log(keypoints[k].size / smallestKPsize)*sizeCst + 0.5), 0);
			if (kpScaleIdx[k] >= nbScales)
				kpScaleIdx[k] = nbScales - 1;

			if (keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || //check if the description at this specific position and scale fits inside the image
				keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.x >= greyImage.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= greyImage.rows - patternSizes[kpScaleIdx[k]]
				)
			{
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}
	else
	{
		const int scIdx = max((int)(1.0986122886681*sizeCst + 0.5), 0);
		for (size_t k = keypoints.size(); k--; )
		{
			kpScaleIdx[k] = scIdx; // equivalent to the formule when the scale is normalized with a constant size of keypoints[k].size=3*SMALLEST_KP_SIZE
			if (kpScaleIdx[k] >= nbScales)
			{
				kpScaleIdx[k] = nbScales - 1;
			}
			if (keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.x >= greyImage.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= greyImage.rows - patternSizes[kpScaleIdx[k]]
				)
			{
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}

	// allocate descriptor memory, estimate orientations, extract descriptors
	if (!extAll)
	{
		// extract the best comparisons only
		_descriptors.create((int)keypoints.size(), CREAKnbPairs / 8, CV_8U);
		_descriptors.setTo(Scalar::all(0));
		Mat descriptors = _descriptors.getMat();

		void *ptr = descriptors.data + (keypoints.size() - 1)*descriptors.step[0];

		for (size_t k = keypoints.size(); k--; )
		{
			for (int r = 0; r < 3; r++)//BGR
			{
				// get the points intensity value in the un-rotated pattern
				for (int i = nbPoints; i--; ) {
					pointsValue[r][i] = meanCREAKIntensity<srcMatType, iiMatType>(splitImage[r], imgIntegral[r],
						keypoints[k].pt.x, keypoints[k].pt.y,
						kpScaleIdx[k], 0, i);
				}
				direction0 = 0;
				direction1 = 0;
				for (int m = CREAKnbOrienpairs; m--; )
				{
					//iterate through the orientation pairs
					const int delta = (pointsValue[r][CREAKorientationPairs[m].i] - pointsValue[r][CREAKorientationPairs[m].j]);
					direction0 += delta * (CREAKorientationPairs[m].weight_dx) / 2048;
					direction1 += delta * (CREAKorientationPairs[m].weight_dy) / 2048;
				}

				keypoints[k].angle = static_cast<float>(atan2((float)direction1, (float)direction0)*(180.0 / CV_PI));//estimate orientation


				if (keypoints[k].angle < 0.f)
					thetaIdx = int(nbOrientation*keypoints[k].angle*(1 / 360.0) - 0.5);
				else
					thetaIdx = int(nbOrientation*keypoints[k].angle*(1 / 360.0) + 0.5);

				if (thetaIdx < 0)
					thetaIdx += nbOrientation;

				if (thetaIdx >= nbOrientation)
					thetaIdx -= nbOrientation;

				// extract descriptor at the computed orientation
				for (int i = nbPoints; i--; ) {
					pointsValue[r][i] = meanCREAKIntensity<srcMatType, iiMatType>(splitImage[r], imgIntegral[r],
						keypoints[k].pt.x, keypoints[k].pt.y,
						kpScaleIdx[k], thetaIdx, i);
				}
			}
			// Extract descriptor
			extractCREAKDescriptor<srcMatType>(pointsValue, &ptr);
		}
	}
	else // extract all possible comparisons for selection
	{
		_descriptors.create((int)keypoints.size(), 128, CV_8U);
		_descriptors.setTo(Scalar::all(0));
		Mat descriptors = _descriptors.getMat();
		bitset<1024>* ptr = (bitset<1024>*) (descriptors.data + (keypoints.size() - 1)*descriptors.step[0]);

		for (size_t k = keypoints.size(); k--; )
		{
			//estimate orientation (gradient)
			if (!orientationNormalized)
			{
				thetaIdx = 0;//assign 0° to all keypoints
				keypoints[k].angle = 0.0;
			}
			else
			{
				for (int r = 0; r < 3; r++)//BGR
				{
					//get the points intensity value in the un-rotated pattern
					for (int i = nbPoints; i--; )
						pointsValue[r][i] = meanCREAKIntensity<srcMatType, iiMatType>(splitImage[r], imgIntegral[r],
							keypoints[k].pt.x, keypoints[k].pt.y,
							kpScaleIdx[k], 0, i);

					direction0 = 0;
					direction1 = 0;
					for (int m = 57; m--; )
					{
						//iterate through the orientation pairs
						const int delta = (pointsValue[r][CREAKorientationPairs[m].i] - pointsValue[r][CREAKorientationPairs[m].j]);
						direction0 += delta * (CREAKorientationPairs[m].weight_dx) / 2048;
						direction1 += delta * (CREAKorientationPairs[m].weight_dy) / 2048;
					}

					keypoints[k].angle = static_cast<float>(atan2((float)direction1, (float)direction0)*(180.0 / CV_PI)); //estimate orientation

					if (keypoints[k].angle < 0.f)
						thetaIdx = int(nbOrientation*keypoints[k].angle*(1 / 360.0) - 0.5);
					else
						thetaIdx = int(nbOrientation*keypoints[k].angle*(1 / 360.0) + 0.5);

					if (thetaIdx < 0)
						thetaIdx += nbOrientation;

					if (thetaIdx >= nbOrientation)
						thetaIdx -= nbOrientation;

					// get the points intensity value in the rotated pattern
					for (int i = nbPoints; i--; )
					{
						pointsValue[r][i] = meanCREAKIntensity<srcMatType, iiMatType>(splitImage[r], imgIntegral[r],
							keypoints[k].pt.x, keypoints[k].pt.y,
							kpScaleIdx[k], thetaIdx, i);
					}
				}
			}
			int cnt(0);
			for (int i = 1; i < nbPoints; ++i)
			{
				//(generate all the pairs)
				for (int j = 0; j < i; ++j)
				{
					//ptr->set(cnt, pointsValue[0][i] >= pointsValue[0][j]); //B
					//ptr->set(cnt, pointsValue[1][i] >= pointsValue[1][j]); //G
					ptr->set(cnt, pointsValue[2][i] >= pointsValue[2][j]); //R
					++cnt;
				}
			}
			--ptr;
		}
	}
}


void computeCREAK(InputArray _image, vector<KeyPoint>& keypoints, OutputArray _descriptors)
{
	mux.lock();
	Mat image = _image.getMat();
	if (image.empty())
		return;
	if (keypoints.empty())
		return;

	buildCREAKPattern();
	computeCREAKDescriptors<uchar, int>(image, keypoints, _descriptors);
	mux.unlock();
}
