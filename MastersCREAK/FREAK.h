#pragma once
#include "FREAK.h"
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
#include <mutex>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

mutex mux;

double patternScale0;
int nOctaves0;
vector<int> selectedPairs0;

bool orientationNormalized = true;
bool scaleNormalized = true;
float patternScale = 22.0f;
int nOctaves = 4;
const double nbLog2 = 0.693147180559945;
const int nbScales = 64;
const int FREAKnbPairs = 512; //Size of descriptor
const int FREAKnbOrienpairs = 45; //CREAK has 57, FREAK has 45
const int nbPoints = 43; //Number of circles
const int nbOrientation = 256;
const int smallestKPsize = 7;

int patternSizes[nbScales]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
struct PatternPoint
{
	float x; // x coordinate relative to center
	float y; // x coordinate relative to center
	float sigma; // Gaussian smoothing sigma (Radius of blur kernel)
};

struct DescriptionPair
{
	uchar i; // index of the first point
	uchar j; // index of the second point
};

struct OrientationPair
{
	uchar i; // index of the first point
	uchar j; // index of the second point
	int weight_dx; // dx/(norm_sq))*4096
	int weight_dy; // dy/(norm_sq))*4096
};

DescriptionPair FREAKdescriptionPairs[FREAKnbPairs];
OrientationPair FREAKorientationPairs[FREAKnbOrienpairs];
vector<PatternPoint> FREAKpatternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and nbOrientation)

//default pairs
const int FREAKdefPairs[FREAKnbPairs] =
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
	236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
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
	670,249,36,581,389,605,331,518,442,822
};
void buildFREAKPattern()
{
	if (patternScale == patternScale0 && nOctaves == nOctaves0 && !FREAKpatternLookup.empty())
		return;

	nOctaves0 = nOctaves;
	patternScale0 = patternScale;

	FREAKpatternLookup.resize(nbScales*nbOrientation*nbPoints);
	double scaleStep = std::pow(2.0, (double)(nOctaves) / nbScales); // 2 ^ ( (nOctaves-1) /nbScales)
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
								radius[3] / 2.0,
								radius[4] / 2.0,
								radius[5] / 2.0,
								radius[6] / 2.0,
								radius[6] / 2.0 };

	// fill the lookup table
	for (int scaleIdx = 0; scaleIdx < nbScales; ++scaleIdx)
	{
		patternSizes[scaleIdx] = 0; // proper initialization
		scalingFactor = std::pow(scaleStep, scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

		for (int orientationIdx = 0; orientationIdx < nbOrientation; ++orientationIdx)
		{
			theta = double(orientationIdx) * 2 * CV_PI / double(nbOrientation); // nbOrientation of the pattern
			int pointIdx = 0;

			PatternPoint* patternLookupPtr = &FREAKpatternLookup[0];
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
	FREAKorientationPairs[0].i = 0; FREAKorientationPairs[0].j = 3; FREAKorientationPairs[1].i = 1; FREAKorientationPairs[1].j = 4; FREAKorientationPairs[2].i = 2; FREAKorientationPairs[2].j = 5;
	FREAKorientationPairs[3].i = 0; FREAKorientationPairs[3].j = 2; FREAKorientationPairs[4].i = 1; FREAKorientationPairs[4].j = 3; FREAKorientationPairs[5].i = 2; FREAKorientationPairs[5].j = 4;
	FREAKorientationPairs[6].i = 3; FREAKorientationPairs[6].j = 5; FREAKorientationPairs[7].i = 4; FREAKorientationPairs[7].j = 0; FREAKorientationPairs[8].i = 5; FREAKorientationPairs[8].j = 1;

	FREAKorientationPairs[9].i = 6; FREAKorientationPairs[9].j = 9; FREAKorientationPairs[10].i = 7; FREAKorientationPairs[10].j = 10; FREAKorientationPairs[11].i = 8; FREAKorientationPairs[11].j = 11;
	FREAKorientationPairs[12].i = 6; FREAKorientationPairs[12].j = 8; FREAKorientationPairs[13].i = 7; FREAKorientationPairs[13].j = 9; FREAKorientationPairs[14].i = 8; FREAKorientationPairs[14].j = 10;
	FREAKorientationPairs[15].i = 9; FREAKorientationPairs[15].j = 11; FREAKorientationPairs[16].i = 10; FREAKorientationPairs[16].j = 6; FREAKorientationPairs[17].i = 11; FREAKorientationPairs[17].j = 7;

	FREAKorientationPairs[18].i = 12; FREAKorientationPairs[18].j = 15; FREAKorientationPairs[19].i = 13; FREAKorientationPairs[19].j = 16; FREAKorientationPairs[20].i = 14; FREAKorientationPairs[20].j = 17;
	FREAKorientationPairs[21].i = 12; FREAKorientationPairs[21].j = 14; FREAKorientationPairs[22].i = 13; FREAKorientationPairs[22].j = 15; FREAKorientationPairs[23].i = 14; FREAKorientationPairs[23].j = 16;
	FREAKorientationPairs[24].i = 15; FREAKorientationPairs[24].j = 17; FREAKorientationPairs[25].i = 16; FREAKorientationPairs[25].j = 12; FREAKorientationPairs[26].i = 17; FREAKorientationPairs[26].j = 13;

	FREAKorientationPairs[27].i = 18; FREAKorientationPairs[27].j = 21; FREAKorientationPairs[28].i = 19; FREAKorientationPairs[28].j = 22; FREAKorientationPairs[29].i = 20; FREAKorientationPairs[29].j = 23;
	FREAKorientationPairs[30].i = 18; FREAKorientationPairs[30].j = 20; FREAKorientationPairs[31].i = 19; FREAKorientationPairs[31].j = 21; FREAKorientationPairs[32].i = 20; FREAKorientationPairs[32].j = 22;
	FREAKorientationPairs[33].i = 21; FREAKorientationPairs[33].j = 23; FREAKorientationPairs[34].i = 22; FREAKorientationPairs[34].j = 18; FREAKorientationPairs[35].i = 23; FREAKorientationPairs[35].j = 19;

	FREAKorientationPairs[36].i = 24; FREAKorientationPairs[36].j = 27; FREAKorientationPairs[37].i = 25; FREAKorientationPairs[37].j = 28; FREAKorientationPairs[38].i = 26; FREAKorientationPairs[38].j = 29;
	FREAKorientationPairs[39].i = 30; FREAKorientationPairs[39].j = 33; FREAKorientationPairs[40].i = 31; FREAKorientationPairs[40].j = 34; FREAKorientationPairs[41].i = 32; FREAKorientationPairs[41].j = 35;
	FREAKorientationPairs[42].i = 36; FREAKorientationPairs[42].j = 39; FREAKorientationPairs[43].i = 37; FREAKorientationPairs[43].j = 40; FREAKorientationPairs[44].i = 38; FREAKorientationPairs[44].j = 41;

	for (unsigned m = FREAKnbOrienpairs; m--; )
	{
		const float dx = FREAKpatternLookup[FREAKorientationPairs[m].i].x - FREAKpatternLookup[FREAKorientationPairs[m].j].x;
		const float dy = FREAKpatternLookup[FREAKorientationPairs[m].i].y - FREAKpatternLookup[FREAKorientationPairs[m].j].y;
		const float norm_sq = (dx*dx + dy * dy);
		FREAKorientationPairs[m].weight_dx = int((dx / (norm_sq))*4096.0 + 0.5);
		FREAKorientationPairs[m].weight_dy = int((dy / (norm_sq))*4096.0 + 0.5);
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

	// Input vector provided
	if (!selectedPairs0.empty())
	{
		if ((int)selectedPairs0.size() == FREAKnbPairs)
		{
			for (int i = 0; i < FREAKnbPairs; ++i)
				FREAKdescriptionPairs[i] = allPairs[selectedPairs0.at(i)];
		}
		else
		{
			CV_Error(Error::StsVecLengthErr, "Input vector does not match the required size");
		}
	}
	else // default selected pairs
	{
		for (int i = 0; i < FREAKnbPairs; ++i)
			FREAKdescriptionPairs[i] = allPairs[FREAKdefPairs[i]];
	}

}

template <typename srcMatType>
void extractFREAKDescriptor(srcMatType *pointsValue, void ** ptr)
{
	std::bitset<FREAKnbPairs>** ptrScalar = (std::bitset<FREAKnbPairs>**) ptr;

	// extracting descriptor preserving the order of SSE version
	/*int cnt = 0;
	for (int n = 7; n < FREAKnbPairs; n += 128)
	{
		for (int m = 8; m--; )
		{
			int nm = n - m;
			for (int kk = nm + 15 * 8; kk >= nm; kk -= 8, ++cnt)
			{
				(*ptrScalar)->set(kk, pointsValue[FREAKdescriptionPairs[cnt].i] >= pointsValue[FREAKdescriptionPairs[cnt].j]);
			}
		}
	}*/
	for (int cnt = 0; cnt < FREAKnbPairs; cnt++)
	{
		(*ptrScalar)->set(cnt, pointsValue[FREAKdescriptionPairs[cnt].i] >= pointsValue[FREAKdescriptionPairs[cnt].j]);
	}
	--(*ptrScalar);
}
/*#if CV_SSE2
void extractFREAKDescriptor(uchar *pointsValue, void ** ptr)
{
	__m128i** ptrSSE = (__m128i**) ptr;

	// note that comparisons order is modified in each block (but first 128 comparisons remain globally the same-->does not affect the 128,384 bits segmanted matching strategy)
	int cnt = 0;
	for (int n = FREAKnbPairs / 128; n--; )
	{
		__m128i result128 = _mm_setzero_si128();
		for (int m = 128 / 16; m--; cnt += 16)
		{
			__m128i operand1 = _mm_set_epi8(pointsValue[FREAKdescriptionPairs[cnt + 0].i],
				pointsValue[FREAKdescriptionPairs[cnt + 1].i],
				pointsValue[FREAKdescriptionPairs[cnt + 2].i],
				pointsValue[FREAKdescriptionPairs[cnt + 3].i],
				pointsValue[FREAKdescriptionPairs[cnt + 4].i],
				pointsValue[FREAKdescriptionPairs[cnt + 5].i],
				pointsValue[FREAKdescriptionPairs[cnt + 6].i],
				pointsValue[FREAKdescriptionPairs[cnt + 7].i],
				pointsValue[FREAKdescriptionPairs[cnt + 8].i],
				pointsValue[FREAKdescriptionPairs[cnt + 9].i],
				pointsValue[FREAKdescriptionPairs[cnt + 10].i],
				pointsValue[FREAKdescriptionPairs[cnt + 11].i],
				pointsValue[FREAKdescriptionPairs[cnt + 12].i],
				pointsValue[FREAKdescriptionPairs[cnt + 13].i],
				pointsValue[FREAKdescriptionPairs[cnt + 14].i],
				pointsValue[FREAKdescriptionPairs[cnt + 15].i]);

			__m128i operand2 = _mm_set_epi8(pointsValue[FREAKdescriptionPairs[cnt + 0].j],
				pointsValue[FREAKdescriptionPairs[cnt + 1].j],
				pointsValue[FREAKdescriptionPairs[cnt + 2].j],
				pointsValue[FREAKdescriptionPairs[cnt + 3].j],
				pointsValue[FREAKdescriptionPairs[cnt + 4].j],
				pointsValue[FREAKdescriptionPairs[cnt + 5].j],
				pointsValue[FREAKdescriptionPairs[cnt + 6].j],
				pointsValue[FREAKdescriptionPairs[cnt + 7].j],
				pointsValue[FREAKdescriptionPairs[cnt + 8].j],
				pointsValue[FREAKdescriptionPairs[cnt + 9].j],
				pointsValue[FREAKdescriptionPairs[cnt + 10].j],
				pointsValue[FREAKdescriptionPairs[cnt + 11].j],
				pointsValue[FREAKdescriptionPairs[cnt + 12].j],
				pointsValue[FREAKdescriptionPairs[cnt + 13].j],
				pointsValue[FREAKdescriptionPairs[cnt + 14].j],
				pointsValue[FREAKdescriptionPairs[cnt + 15].j]);

			__m128i workReg = _mm_min_epu8(operand1, operand2); // emulated "not less than" for 8-bit UNSIGNED integers
			workReg = _mm_cmpeq_epi8(workReg, operand2);        // emulated "not less than" for 8-bit UNSIGNED integers

			workReg = _mm_and_si128(_mm_set1_epi16(short(0x8080 >> m)), workReg); // merge the last 16 bits with the 128bits std::vector until full
			result128 = _mm_or_si128(result128, workReg);
		}
		(**ptrSSE) = result128;
		++(*ptrSSE);
	}
	(*ptrSSE) -= 8;
}
#endif*/

template <typename imgType, typename iiType>
imgType meanFREAKIntensity(InputArray _image, InputArray _integral,
	const float kp_x,
	const float kp_y,
	const unsigned int scale,
	const unsigned int rot,
	const unsigned int point)
{
	Mat image = _image.getMat(), integral = _integral.getMat();
	// get point position in image
	const PatternPoint& FreakPoint = FREAKpatternLookup[scale*nbOrientation*nbPoints + rot * nbPoints + point];
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

	// expected case:

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
	//~ std::cout<<integral.step[1]<<std::endl;
	return static_cast<imgType>(ret_val);
}

template <typename srcMatType, typename iiMatType>
void computeFREAKDescriptors(InputArray _image, vector<KeyPoint>& keypoints, OutputArray _descriptors)
{
	Mat image = _image.getMat();
	Mat imgIntegral;
	integral(image, imgIntegral, DataType<iiMatType>::type);
	vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
	const vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in std::vector erase function
	const vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
	const float sizeCst = static_cast<float>(nbScales / (nbLog2* nOctaves));
	srcMatType pointsValue[nbPoints];
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
				keypoints[k].pt.x >= image.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= image.rows - patternSizes[kpScaleIdx[k]]
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
				keypoints[k].pt.x >= image.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= image.rows - patternSizes[kpScaleIdx[k]]
				)
			{
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}

	// allocate descriptor memory, estimate orientations, extract descriptors
	// extract the best comparisons only
	_descriptors.create((int)keypoints.size(), FREAKnbPairs / 8, CV_8U);
	_descriptors.setTo(Scalar::all(0));
	Mat descriptors = _descriptors.getMat();

	void *ptr = descriptors.data + (keypoints.size() - 1)*descriptors.step[0];

	for (size_t k = keypoints.size(); k--; ) {
		// estimate orientation (gradient)
		if (!orientationNormalized)
		{
			thetaIdx = 0; // assign 0° to all keypoints
			keypoints[k].angle = 0.0;
		}
		else
		{
			// get the points intensity value in the un-rotated pattern
			for (int i = nbPoints; i--; ) {
				pointsValue[i] = meanFREAKIntensity<srcMatType, iiMatType>(image, imgIntegral,
					keypoints[k].pt.x, keypoints[k].pt.y,
					kpScaleIdx[k], 0, i);
			}
			direction0 = 0;
			direction1 = 0;
			for (int m = FREAKnbOrienpairs; m--; )
			{
				//iterate through the orientation pairs
				const int delta = (pointsValue[FREAKorientationPairs[m].i] - pointsValue[FREAKorientationPairs[m].j]);
				direction0 += delta * (FREAKorientationPairs[m].weight_dx) / 2048;
				direction1 += delta * (FREAKorientationPairs[m].weight_dy) / 2048;
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
		}
		// extract descriptor at the computed orientation
		for (int i = nbPoints; i--; ) {
			pointsValue[i] = meanFREAKIntensity<srcMatType, iiMatType>(image, imgIntegral,
				keypoints[k].pt.x, keypoints[k].pt.y,
				kpScaleIdx[k], thetaIdx, i);
		}
		// Extract descriptor
		extractFREAKDescriptor<srcMatType>(pointsValue, &ptr);
	}
}

void computeFREAK(InputArray _image, vector<KeyPoint>& keypoints, OutputArray _descriptors)
{
	mux.lock();
	Mat image = _image.getMat();
	if (image.empty())
		return;
	if (keypoints.empty())
		return;

	buildFREAKPattern();

	//CREAK uses colour image
	Mat grayImage;// = image;
	// Convert to gray if not already
	if (image.channels() == 3 || image.channels() == 4)
		cvtColor(image, grayImage, COLOR_BGR2GRAY);
	else {
		CV_Assert(image.channels() == 1);
		grayImage = image;
	}

	// Use 32-bit integers if we won't overflow in the integral image
	if ((image.depth() == CV_8U || image.depth() == CV_8S) &&
		(image.rows * image.cols) < 8388608) // 8388608 = 2 ^ (32 - 8(bit depth) - 1(sign bit))
	{
		// Create the integral image appropriate for our type & usage
		if (image.depth() == CV_8U)
			computeFREAKDescriptors<uchar, int>(grayImage, keypoints, _descriptors);
		else if (image.depth() == CV_8S)
			computeFREAKDescriptors<char, int>(grayImage, keypoints, _descriptors);
		else
			CV_Error(Error::StsUnsupportedFormat, "");
	}
	else {
		// Create the integral image appropriate for our type & usage
		if (image.depth() == CV_8U)
			computeFREAKDescriptors<uchar, double>(grayImage, keypoints, _descriptors);
		else if (image.depth() == CV_8S)
			computeFREAKDescriptors<char, double>(grayImage, keypoints, _descriptors);
		else if (image.depth() == CV_16U)
			computeFREAKDescriptors<ushort, double>(grayImage, keypoints, _descriptors);
		else if (image.depth() == CV_16S)
			computeFREAKDescriptors<short, double>(grayImage, keypoints, _descriptors);
		else
			CV_Error(Error::StsUnsupportedFormat, "");
	}
	mux.unlock();
}
