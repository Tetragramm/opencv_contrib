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

namespace cv
{
namespace mapping3d
{

/**
 * Position Calculation
 */
    class PositionCalculatorImpl : public PositionCalculator
    {
    protected:
        virtual bool computeStateImpl(double time, OutputArray _state, OutputArray _covariance = noArray());
        virtual void addMeasurementImpl( InputArray _tvec, InputArray _rvec, const Point2f _pt, double time,
                                         InputArray _cameraMatrix, InputArray _distortionMatrix );
        std::deque<Mat> positions;
        std::deque<Point2d> angles;
        Mat F, G, b, W;
    };

    bool PositionCalculatorImpl::computeStateImpl(double time, OutputArray _state, OutputArray _covariance)
    {
        _state.create(3, 1, CV_64F);
        Mat state = _state.getMat();

        int num = (int)positions.size();
        F.create(2 * num, 3, CV_64F);
        b.create(2 * num, 1, CV_64F);

        for (int i = 0; i < num; ++i)
        {
            F.at<double>(i, 0) = sin(angles[i].x);
            F.at<double>(i, 1) = -cos(angles[i].x);
            F.at<double>(i, 2) = 0;

            F.at<double>(num + i, 0) = -sin(angles[i].y) * cos(angles[i].x);
            F.at<double>(num + i, 1) = -sin(angles[i].y) * sin(angles[i].x);
            F.at<double>(num + i, 2) = cos(angles[i].y);

            b.at<double>(i) = (F(Rect(0, i, 3, 1))*positions[i]).operator cv::Mat().at<double>(0);
            b.at<double>(num + i) = (F(Rect(0, num + i, 3, 1))*positions[i]).operator cv::Mat().at<double>(0);
        }

        Mat iple;
        solve(F, b, iple, DECOMP_SVD);

        Point2d newAng;
        G.create(2 * num, 3, CV_64F);
        W.create(2 * num, 2 * num, CV_64F);
        W.setTo(0);

        for (int i = 0; i < num; ++i)
        {
            newAng = getAzEl(positions[i], iple);
            G.at<double>(i, 0) = sin(newAng.x);
            G.at<double>(i, 1) = -cos(newAng.x);
            G.at<double>(i, 2) = 0;

            G.at<double>(num + i, 0) = -sin(newAng.y) * cos(newAng.x);
            G.at<double>(num + i, 1) = -sin(newAng.y) * sin(newAng.x);
            G.at<double>(num + i, 2) = cos(newAng.y);

            Point2f diff2;
            Point3f diff3;

            diff2.x = (iple.at<double>(0) - positions[i].at<double>(0));
            diff2.y = (iple.at<double>(1) - positions[i].at<double>(1));
            diff3.x = diff2.x;
            diff3.y = diff2.y;
            diff3.z = (iple.at<double>(2) - positions[i].at<double>(2));
            W.at<double>(i, i) = norm(diff2);
            W.at<double>(i + num, i + num) = (norm(diff3) - W.at<double>(i, i));
        }
        
        state = (G.t()*W*F).operator cv::Mat().inv()*G.t()*W*b;

        if (_covariance.needed())
        {
            Mat& cov = _covariance.getMatRef();
            cov = G.t()*F / (num / 2.0);
            multiply(cov, sum(F*state - b)(0), cov);
        }
        return true;
    }

    void PositionCalculatorImpl::addMeasurementImpl( InputArray _tvec, InputArray _rvec, const Point2f _pt, double time,
                                                     InputArray _cameraMatrix, InputArray _distortionMatrix )
    {
        Mat tvec = _tvec.getMat();
        Mat rvec = _rvec.getMat();
        Mat cameraMatrix = _cameraMatrix.getMat();
        Mat distortionMatrix = _distortionMatrix.getMat();

        std::vector<Point2f> ptsIn, ptsOut;
        ptsIn.push_back(_pt);
        undistortPoints(ptsIn, ptsOut, cameraMatrix, distortionMatrix, noArray(), cameraMatrix);

        double x = ptsOut[0].x;
        double y = ptsOut[0].y;

        Mat LOS(3,1,CV_64F);

        LOS.at<double>(0) = ptsOut[0].x;
        LOS.at<double>(1) = ptsOut[0].y;
        LOS.at<double>(2) = 1;
        
        LOS = cameraMatrix.inv()*LOS;

        Mat cameraRotation, cameraTranslation;
        Rodrigues(rvec, cameraRotation);
        cameraRotation = cameraRotation.t();
        cameraTranslation = (-cameraRotation * tvec);
        LOS = cameraRotation*LOS;

        positions.push_back(cameraTranslation);
        Mat zeroPos(3, 1, CV_64F);
        zeroPos.setTo(0);
        Point2d azel = getAzEl(zeroPos, LOS);
        angles.push_back(azel);
    }

    Ptr<PositionCalculator> PositionCalculator::create()
    {
        return Ptr<PositionCalculatorImpl>(new PositionCalculatorImpl());
    }

	void calcPosition( InputArray _tvecs, InputArray _rvecs, InputArray _pts,
					   InputArray _cameraMatrices, InputArray _distortionMatrices,
					   OutputArray _state, OutputArray _covariance )
	{
		Ptr<PositionCalculator> pPC = PositionCalculator::create();

		std::vector<Mat> tvecs, rvecs;
		_tvecs.getMatVector( tvecs );
		_rvecs.getMatVector( rvecs );

		CV_Assert( tvecs.size() >= 2 );
		CV_Assert( tvecs.size() == rvecs.size() );

		Mat pts = _pts.getMat();

		CV_Assert( ( tvecs.size() == pts.checkVector( 2, CV_32F, true ) ) );
		
		std::vector<Mat> cameraM, distM;
		if( _cameraMatrices.kind() == _InputArray::STD_VECTOR_MAT )
		{
			_cameraMatrices.getMatVector( cameraM );
			CV_Assert( tvecs.size() == cameraM.size() );
		}
		else
		{
			cameraM.push_back( _cameraMatrices.getMat() );
			CV_Assert( ( cameraM[0].rows == 3 ) && ( cameraM[0].cols == 3 ) );
		}

		if( _distortionMatrices.kind() == _InputArray::STD_VECTOR_MAT )
		{
			_distortionMatrices.getMatVector( distM );
			CV_Assert( tvecs.size() == distM.size() );
		}
		else
		{
			distM.push_back( _distortionMatrices.getMat() );
			CV_Assert( ( ( distM[0].rows == 5 ) && ( distM[0].cols == 1 ) ) || distM[0].empty() );
		}

		Mat camera = cameraM[0];
		Mat dist = distM[0];
		for( size_t i = 0; i < tvecs.size(); ++i )
		{
			if( cameraM.size() == tvecs.size() )
				camera = cameraM[i];
			if( distM.size() == tvecs.size() )
				dist = distM[i];
			pPC->addMeasurement( tvecs[i], rvecs[i], pts.at<Point2f>( i ), camera, dist );
		}
		pPC->computeState( _state, _covariance );
	}

}/* namespace mapping3d */
}/* namespace cv */
