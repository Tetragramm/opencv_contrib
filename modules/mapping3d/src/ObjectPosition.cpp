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
// Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"
#include "mappingFuncs.hpp"
#include <iostream>

namespace cv
{
	namespace mapping3d
	{
		void findCentroid( InputArray pointsA, OutputArray centroid )
		{
			using std::vector;
			centroid.create( 1, 3, CV_64F );
			Mat& centroid_state = centroid.getMatRef();
			centroid_state.setTo( 0 );
			vector<Mat> states;
			pointsA.getMatVector( states );

			int numPoints = pointsA.total();
			for( unsigned int i = 0; i < numPoints; i++ )
			{
				add( centroid_state, states[i], centroid_state );
			}
			multiply( centroid_state, 1.0 / numPoints, centroid_state );
		}

		void findRotationAndTranslation( InputArray pointsFound, InputArray pointsModel,
										 InputArray centroidFound, InputArray centroidModel,
										 OutputArray tvec, OutputArray rvec )
		{
			using std::vector;
			Mat centroid_state = centroidFound.getMat();
			Mat centroid_obj = centroidModel.getMat();

			vector<Mat> states;
			pointsFound.getMatVector( states );
			vector<Mat> objPoints;
			pointsModel.getMatVector( objPoints );

			int numPoints = pointsFound.total();
			Mat H( 3, 3, CV_64F );
			H.setTo( 0 );
			for( unsigned int i = 0; i < numPoints; i++ )
			{
				Mat centered_state, centered_obj;
				subtract( states[i], centroid_state, centered_state );
				subtract( objPoints[i], centroid_obj, centered_obj );
				Mat test = centered_obj.t()*centered_state;
				add( H, test, H );
			}

			Mat U, S, V;
			SVDecomp( H, S, U, V );
			multiply( V.row( 0 ), -1, V.row( 0 ) );
			multiply( V.row( 2 ), -1, V.row( 2 ) );
			multiply( U.col( 0 ), -1, U.col( 0 ) );
			Mat R = V.t()*U.t();
			if( determinant( R ) < 0.0 )
			{
				multiply( R.col( 2 ), -1, R.col( 2 ) );
			}
			Rodrigues( R, rvec );

			tvec.create( 3, 1, CV_64F );
			Mat& t = tvec.getMatRef();
			Mat temp = (-R*centroid_obj.t());
			add( temp, centroid_state.t(), t );
		}

		void calcObjectPosition( InputArrayOfArrays imagePointsPerView, InputArray objectPoints,
								 InputArray _cameraMatrices, InputArray _distortionMatrices,
								 InputArray _tvecs, InputArray _rvecs,
								 OutputArray tvecObject, OutputArray rvecObject )
		{
			using std::vector;

			Mat centroid_state;
			Mat centroid_obj;
			vector<Mat> states;
			vector<Mat> objPoints;
			objectPoints.getMatVector( objPoints );

		    const unsigned int numPoints = imagePointsPerView.total();
			for( unsigned int i = 0; i < numPoints; i++ )
			{
				Mat points_mat = imagePointsPerView.getMat( i );

				Mat state;
				calcPosition( _tvecs, _rvecs, points_mat, _cameraMatrices, _distortionMatrices, state );
				states.push_back( state );
			}

			register3dPoints( states, objPoints, tvecObject, rvecObject );
		}

		void register3dPoints(InputArray pointsFound, InputArray pointsModel,
							   OutputArray tvec, OutputArray rvec)
		{
			Mat centroid_state;
			Mat centroid_obj;

			findCentroid( pointsFound, centroid_state );
			findCentroid( pointsModel, centroid_obj );

			findRotationAndTranslation( pointsFound, pointsModel, centroid_state, centroid_obj, tvec, rvec );
		}
	}/* namespace mapping3d */
}/* namespace cv */
