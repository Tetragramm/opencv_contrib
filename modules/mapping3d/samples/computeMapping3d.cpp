/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2017, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/core/utility.hpp>
#include <opencv2/mapping3d.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace mapping3d;
using namespace chrono;

static const char* keys =
{ "{@sample_path   |1| Path of the folder containing the sample files}" };

static void help()
{
	cout << "\nThis example shows the functionality of \"Mapping3d \""
		"Call:\n"
		"./example_mapping3d_computeMapping3d <sample_path> \n"
		<< endl;
}

void readImagesAndExtrinsics(String sample_path, vector<Mat>& images, vector<Mat>& rBuffer, vector<Mat>& tBuffer, Mat& cameraMatrix)
{
	//Open the xml file containing the camera intrinsic and extrinsics.
	FileStorage fs;
	fs.open( sample_path+"Poses.xml", FileStorage::FORMAT_AUTO );

	//Read in the camera matrix
	fs["Camera_Matrix"] >> cameraMatrix;
	cameraMatrix.convertTo( cameraMatrix, CV_64F );

	//Read in the number of frames
	int numFrames;
	fs["nr_of_frames"] >> numFrames;

	for( int i = 1, j = 0; i < numFrames; i += 10, ++j )
	{
		std::stringstream str1, str2;
		str1 << sample_path << i << ".png";
		str2 << "Pose_Matrix_" << i;
		cout << "Reading File " << i << ".png" << "\n";
		//Create the File path and read in the image
		images.push_back( imread( str1.str() ) );
		Mat pose, rvec;
		//Read in the OpenGL Pose
		fs[str2.str()] >> pose;

		//Convert the OpenGL Pose to the OpenCV system
		//A few operations here can probably be removed,
		//but this isn't the point of the example.
		//I had it running through the VIZ module for visualization,
		//and that requires stopping in the middle because it uses yet
		//ANOTHER coordinate system.
		Mat ident;
		ident.create( 3, 3, pose.type() );
		setIdentity( ident, -1 );
		ident.at<float>( 0, 0 ) = 1;

		pose( Rect( 0, 0, 3, 3 ) ) = ( ident * pose( Rect( 0, 0, 3, 3 ) ).t() ).t();

		Rodrigues( pose( Rect( 0, 0, 3, 3 ) ), rvec );
		rBuffer.push_back( rvec.clone() );
		tBuffer.push_back( pose( Rect( 3, 0, 1, 3 ) ).clone() );

		Mat R;
		Rodrigues( rBuffer[j], R );
		R = R.t();
		tBuffer[j] = ( -R * tBuffer[j] );
		Rodrigues( R, rBuffer[j] );

		//Convert to the double type.
		tBuffer[j].convertTo( tBuffer[j], CV_64F );
		rBuffer[j].convertTo( rBuffer[j], CV_64F );
	}
}

void writePLY( Mat PC, const char* FileName )
{
	std::ofstream outFile( FileName );

	if( !outFile )
	{
		//cerr << "Error opening output file: " << FileName << "!" << endl;
		printf( "Error opening output file: %s!\n", FileName );
		exit( 1 );
	}

	////
	// Header
	////

	const int pointNum = (int) PC.rows;
	const int vertNum = (int) PC.cols;

	outFile << "ply" << std::endl;
	outFile << "format ascii 1.0" << std::endl;
	outFile << "element vertex " << pointNum << std::endl;
	outFile << "property float x" << std::endl;
	outFile << "property float y" << std::endl;
	outFile << "property float z" << std::endl;
	if( vertNum == 6 )
	{
		outFile << "property float nx" << std::endl;
		outFile << "property float ny" << std::endl;
		outFile << "property float nz" << std::endl;
	}
	outFile << "end_header" << std::endl;

	////
	// Points
	////

	for( int pi = 0; pi < pointNum; ++pi )
	{
		const float* point = PC.ptr<float>( pi );

		outFile << point[0] << " " << point[1] << " " << point[2];

		if( vertNum == 6 )
		{
			outFile << " " << point[3] << " " << point[4] << " " << point[5];
		}

		outFile << std::endl;
	}
}

int main( int argc, char** argv )
{
	CommandLineParser parser( argc, argv, keys );

	String sample_path = parser.get<String>( 0 );

	if( sample_path.empty() )
	{
	help();
	return -1;
	}

	//Declare the vectors to hold the images and extrinsics
	vector<Mat> imgBuffer, rBuffer, tBuffer;
    Mat cameraMatrix;
	readImagesAndExtrinsics( sample_path, imgBuffer, rBuffer, tBuffer, cameraMatrix );

	//These get re-used many times.  Save the allocations.
	vector<Mat> keyFramePyr, dstFramePyr;

	//Declare the output variable
	Mat state, PCL( 1, 3, CV_64F );
	PCL.setTo( 0 );

	//Timing variables
	long long timeTotal = 0;
	long long ptsTotal = 0;

	//For each image...
	for( int i = 0; i < imgBuffer.size(); i += 1 )
	{
		//Declare variables and ORB detector
		vector<Point2f> keyFramePts, dstFramePts;
		vector<vector<Point2f>> trackingPts;
		Ptr<ORB> pORB = ORB::create( 10000, 1.2, 1 );
		vector<KeyPoint> kps;

		//Detect keypoints in the image
		pORB->detect( imgBuffer[i], kps );

		//Start track of the poses
		vector<Mat> localt;
		vector<Mat> localr;
		localt.push_back( tBuffer[i] );
		localr.push_back( rBuffer[i] );

		//Save all the key points detected
		for( KeyPoint& p : kps )
		{
			keyFramePts.push_back( p.pt );
			trackingPts.push_back( vector<Point2f>( 1, p.pt ) );
		}

		//Build Optical flow pyramid to save time
		buildOpticalFlowPyramid( imgBuffer[i], keyFramePyr, Size( 11, 11 ), 3 );
		
		//For every other image...
		for( int j = 0; j < imgBuffer.size(); j++ )
		{
			if( j != i )
			{
				//Save the pose
				localt.push_back( tBuffer[j] );
				localr.push_back( rBuffer[j] );

				//And track the points from the outer loop image to the inner loop image
				Mat status, err;
				buildOpticalFlowPyramid( imgBuffer[j], dstFramePyr, Size( 11, 11 ), 3 );
				if( dstFramePts.size() == keyFramePts.size() )
					cv::calcOpticalFlowPyrLK( keyFramePyr, dstFramePyr, keyFramePts, dstFramePts, status, err, Size( 11, 11 ), 3, TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01 ), OPTFLOW_USE_INITIAL_FLOW );
				else
					cv::calcOpticalFlowPyrLK( keyFramePyr, dstFramePyr, keyFramePts, dstFramePts, status, err, Size( 11, 11 ), 3 );
				
				//For every point that was sucessfully tracked, save it.
				//Otherwise remove that point from the saved list.
				for( int idx = 0, idx2 = 0; idx < keyFramePts.size(); ++idx, ++idx2 )
				{
					if( status.at<uchar>( idx2 ) == 1 )
						trackingPts[idx].push_back( dstFramePts[idx2] );
					else
					{
						trackingPts.erase( trackingPts.begin() + idx );
						keyFramePts.erase( keyFramePts.begin() + idx );
						idx--;
					}
				}
			}
		}

		auto start = high_resolution_clock::now();
		for( int idx = 0; idx < trackingPts.size(); ++idx )
		{
			//Using the list of poses, and one set of points (it's location in each frame)
			//along with the camera intrinsics, calculate the 3d position and uncertainty.
			Mat cov;
			mapping3d::calcPosition( localt, localr, trackingPts[idx], cameraMatrix, noArray(), state, cov );
			//If the worst uncertainty is small, add it to the point cloud, otherwise erase it.
			double min, max;
			minMaxIdx( cov, &min, &max );
			if( MAX( abs( max ), abs( min ) ) < 0.003 )
				vconcat( PCL, state.t(), PCL );
			else
			{
				trackingPts.erase( trackingPts.begin() + idx );
				keyFramePts.erase( keyFramePts.begin() + idx );
				idx--;
			}
		}
		auto stop = high_resolution_clock::now();
		timeTotal += duration_cast<nanoseconds>( stop - start ).count();
		ptsTotal += trackingPts.size();
	}
    cout << timeTotal / 1.0e9 << " seconds for " << ptsTotal << " points through "
            << imgBuffer.size() << " frames.\n";
    cout << "This excludes the time spent on optical flow and is just the time calculating 3d positions.";

	//Reshape the Point Cloud Matrix and print it to a PLY file.
	PCL = PCL.reshape( 3, PCL.rows );
	string fileName = sample_path + "Couch_Cloud.ply";
	writePLY( PCL, fileName.c_str() );

	return 0;
}
