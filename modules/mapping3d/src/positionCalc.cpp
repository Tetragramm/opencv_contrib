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
            bool computeStateImpl( double time, OutputArray state, OutputArray covariance = noArray() ) override;
            void addMeasurementImpl( InputArray tvec, InputArray _rvec, const Point2f pt, double time,
                                     InputArray _cameraMatrix, InputArray _distortionMatrix ) override;
            std::deque< Mat > positions;
            std::deque< Point2d > angles;
            Mat F, G, b, W;
        };

        bool PositionCalculatorImpl::computeStateImpl( double /*time*/, OutputArray _state, OutputArray _covariance )
        {
            _state.create( 3, 1, CV_64F );
            Mat& state = _state.getMatRef();

            const int num = static_cast< int >(positions.size());
            F.create( 2 * num, 3, CV_64F );
            b.create( 2 * num, 1, CV_64F );

            for ( int i = 0; i < num; ++i )
            {
                F.at< double >( i, 0 ) = sin( angles[i].x );
                F.at< double >( i, 1 ) = -cos( angles[i].x );
                F.at< double >( i, 2 ) = 0;

                F.at< double >( num + i, 0 ) = -sin( angles[i].y ) * cos( angles[i].x );
                F.at< double >( num + i, 1 ) = -sin( angles[i].y ) * sin( angles[i].x );
                F.at< double >( num + i, 2 ) = cos( angles[i].y );

                b.at< double >( i ) = ( F( Rect( 0, i, 3, 1 ) ) * positions[i] ).operator Mat().at< double >( 0 );
                b.at< double >( num + i ) = ( F( Rect( 0, num + i, 3, 1 ) ) * positions[i] ).operator Mat().at< double
                >( 0 );
            }

            Mat iple;
            solve( F, b, iple, DECOMP_SVD );

            G.create( 2 * num, 3, CV_64F );
            W.create( 2 * num, 2 * num, CV_64F );
            W.setTo( 0 );

            for ( int i = 0; i < num; ++i )
            {
                const Point2d new_ang = getAzEl( positions[i], iple );
                G.at< double >( i, 0 ) = sin( new_ang.x );
                G.at< double >( i, 1 ) = -cos( new_ang.x );
                G.at< double >( i, 2 ) = 0;

                G.at< double >( num + i, 0 ) = -sin( new_ang.y ) * cos( new_ang.x );
                G.at< double >( num + i, 1 ) = -sin( new_ang.y ) * sin( new_ang.x );
                G.at< double >( num + i, 2 ) = cos( new_ang.y );

                Point2f diff2;
                Point3f diff3;

                diff2.x = ( iple.at< double >( 0 ) - positions[i].at< double >( 0 ) );
                diff2.y = ( iple.at< double >( 1 ) - positions[i].at< double >( 1 ) );
                diff3.x = diff2.x;
                diff3.y = diff2.y;
                diff3.z = ( iple.at< double >( 2 ) - positions[i].at< double >( 2 ) );
                W.at< double >( i, i ) = norm( diff2 );
                W.at< double >( i + num, i + num ) = ( norm( diff3 ) - W.at< double >( i, i ) );
            }

            state = ( G.t() * W * F ).operator Mat().inv() * G.t() * W * b;

            if ( _covariance.needed() )
            {
                Mat& cov = _covariance.getMatRef();
                cov = G.t() * F / ( num / 2.0 );
                multiply( cov, sum( F * state - b )( 0 ), cov );
            }
            return true;
        }

        void PositionCalculatorImpl::addMeasurementImpl( InputArray _tvec, InputArray _rvec, const Point2f _pt
                                                         , double /*time*/,
                                                         InputArray _cameraMatrix, InputArray _distortionMatrix )
        {
            Mat tvec = _tvec.getMat();
            Mat rvec = _rvec.getMat();
            Mat camera_matrix = _cameraMatrix.getMat();
            const Mat distortion_matrix = _distortionMatrix.getMat();

            std::vector< Point2f > pts_in, pts_out;
            pts_in.push_back( _pt );
            undistortPoints( pts_in, pts_out, camera_matrix, distortion_matrix, noArray(), camera_matrix );

            Mat los( 3, 1,CV_64F );

            los.at< double >( 0 ) = pts_out[0].x;
            los.at< double >( 1 ) = pts_out[0].y;
            los.at< double >( 2 ) = 1;

            if ( camera_matrix.type() != CV_64F )
                camera_matrix.convertTo( camera_matrix, CV_64F );
            if ( rvec.type() != CV_64F )
                rvec.convertTo( rvec, CV_64F );
            if ( tvec.type() != CV_64F )
                tvec.convertTo( tvec, CV_64F );

            los = camera_matrix.inv() * los;

            Mat camera_rotation;
            if ( rvec.rows == 3 && rvec.cols == 3 )
                camera_rotation = rvec;
            else
                Rodrigues( rvec, camera_rotation );

            if(tvec.rows == 1)
                tvec = tvec.t();

            camera_rotation = camera_rotation.t();
            const Mat camera_translation = ( -camera_rotation * tvec );
            los = camera_rotation * los;

            positions.push_back( camera_translation );
            Mat zero_pos( 3, 1, CV_64F );
            zero_pos.setTo( 0 );
            const Point2d azel = getAzEl( zero_pos, los );
            angles.push_back( azel );
        }

        Ptr< PositionCalculator > PositionCalculator::create()
        {
            return Ptr< PositionCalculatorImpl >( new PositionCalculatorImpl() );
        }

        void calcPosition( InputArray _tvecs, InputArray _rvecs, InputArray _pts,
                           InputArray _cameraMatrices, InputArray _distortionMatrices,
                           OutputArray _state, OutputArray _covariance )
        {
            Ptr< PositionCalculator > p_pc = PositionCalculator::create();

            std::vector< Mat > tvecs, rvecs;
            _tvecs.getMatVector( tvecs );
            _rvecs.getMatVector( rvecs );

            CV_Assert( tvecs.size() >= 2 );
            CV_Assert( tvecs.size() == rvecs.size() );

            Mat pts = _pts.getMat();

            CV_Assert( ( tvecs.size() == pts.checkVector( 2, CV_32F, true ) ) );

            std::vector< Mat > camera_m, dist_m;
            if ( _cameraMatrices.kind() == _InputArray::STD_VECTOR_MAT )
            {
                _cameraMatrices.getMatVector( camera_m );
                CV_Assert( tvecs.size() == camera_m.size() );
            }
            else
            {
                camera_m.push_back( _cameraMatrices.getMat() );
                CV_Assert( ( camera_m[0].rows == 3 ) && ( camera_m[0].cols == 3 ) );
            }

            if ( _distortionMatrices.kind() == _InputArray::STD_VECTOR_MAT )
            {
                _distortionMatrices.getMatVector( dist_m );
                CV_Assert( tvecs.size() == dist_m.size() );
            }
            else
            {
                dist_m.push_back( _distortionMatrices.getMat() );
                CV_Assert( ( ( dist_m[0].rows == 5 ) && ( dist_m[0].cols == 1 ) ) || dist_m[0].empty() );
            }

            Mat camera = camera_m[0];
            Mat dist = dist_m[0];
            for ( size_t i = 0; i < tvecs.size(); ++i )
            {
                if ( camera_m.size() == tvecs.size() )
                    camera = camera_m[i];
                if ( dist_m.size() == tvecs.size() )
                    dist = dist_m[i];
                p_pc->addMeasurement( tvecs[i], rvecs[i], pts.at< Point2f >( i ), camera, dist );
            }
            p_pc->computeState( _state, _covariance );
        }
    }/* namespace mapping3d */
}/* namespace cv */
