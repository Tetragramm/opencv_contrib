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

#ifndef __OPENCV_MAPPING3D_BASE_CLASSES_HPP__
#define __OPENCV_MAPPING3D_BASE_CLASSES_HPP__

#include "opencv2/core.hpp"
#include "opencv2/cvconfig.h"
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <complex>
#include <deque>


namespace cv
{
namespace mapping3d
{

//! @addtogroup saliency
//! @{

/************************************ Saliency Base Class ************************************/

class CV_EXPORTS StateCalculator : public virtual Algorithm
{
 public:
  /**
   * \brief Destructor
   */
  virtual ~StateCalculator();

  /**
   * \brief Create Saliency by saliency type.
   */
  static Ptr<StateCalculator> create( const String& calculationType );

  /**
   * \brief Compute the state
   * \param time        The current time.
   * \param _state      The computed state of the point.
   * \return true if the state is computed, false otherwise
   */
  bool computeState( double time, OutputArray _state, OutputArray _covariance = noArray() );

  /**
   * \brief Get the name of the specific calculation type
   * \return The name of the class
   */
  String getClassName() const;

  void addMeasurement( InputArray _tvec, InputArray _rvec, const Point2f _pt, double time,
					   InputArray _cameraMatrix, InputArray _distortionMatrix );

  void setLimit(size_t _limit);

  size_t getLimit() const;

 protected:
     size_t limit;

  virtual bool computeStateImpl( double time, OutputArray _state, OutputArray _covariance = noArray() ) = 0;
  virtual void addMeasurementImpl( InputArray _tvec, InputArray _rvec, const Point2f _pt, double time,
                                   InputArray _cameraMatrix, InputArray _distortionMatrix ) = 0;

  String className;
};

/************************************ Static Saliency Base Class ************************************/
class CV_EXPORTS PositionCalculator : public virtual StateCalculator
{
 public:

     /** @brief Calculates the 3D position of a point

     This class calculates the combination of measurements from one or more cameras to into the 3D position of a point based upon the paper 3DTMA.

     @param _tvecs Vector of camera translation vectors, ie. from estimatePose.
     @param _rvecs Vector of camera rotation angles, ie. from estimatePose.  Same length as _tvecs.
     @param _pts Vector of 2D object locations within the images, Point2f format, Same length as _tvecs.
     @param _cameraMatrices One Mat or Vector of camera matrices, ie. from calibrateCamera
     @param _distortionMatrices One Mat or Vector of distortion matrices, ie. from calibrateCamera
     @param _location the 3D location of the object, relative to (0,0,0), a Mat.

     @sa  calibrateCamera PositionCalculator
     */

     /**
     * \brief Compute the state
     * \param _state      The computed state of the point.
     * \return true if the state is computed, false otherwise
     */
     bool computeState(OutputArray _state, OutputArray _covariance = noArray());

	 void addMeasurement( InputArray _tvec, InputArray _rvec, const Point2f _pt,
						  InputArray _cameraMatrix, InputArray _distortionMatrix );

     static Ptr<PositionCalculator> create();

protected:
  virtual bool computeStateImpl(double time, OutputArray _state, OutputArray _covariance = noArray()) = 0;
  virtual void addMeasurementImpl( InputArray _tvec, InputArray _rvec, const Point2f _pt, double time,
                                   InputArray _cameraMatrix, InputArray _distortionMatrix ) = 0;

};

class CV_EXPORTS VelocityCalculator : public virtual StateCalculator
{
public:

    /** @brief Calculates the 3D position of a point

    This class calculates the combination of measurements from one or more cameras to into the 3D position of a point based upon the paper 3DTMA.

    @param _tvecs Vector of camera translation vectors, ie. from estimatePose.
    @param _rvecs Vector of camera rotation angles, ie. from estimatePose.  Same length as _tvecs.
    @param _pts Vector of 2D object locations within the images, Point2f format, Same length as _tvecs.
    @param _cameraMatrices One Mat or Vector of camera matrices, ie. from calibrateCamera
    @param _distortionMatrices One Mat or Vector of distortion matrices, ie. from calibrateCamera
    @param _location the 3D location of the object, relative to (0,0,0), a Mat.

    @sa  calibrateCamera PositionCalculator
    */

    static Ptr<VelocityCalculator> create();

protected:
    virtual bool computeStateImpl(double time, OutputArray _state, OutputArray _covariance = noArray()) = 0;
    virtual void addMeasurementImpl( InputArray _tvec, InputArray _rvec, const Point2f _pt, double time,
                                     InputArray _cameraMatrix, InputArray _distortionMatrix ) = 0;
};

//! @}

} /* namespace mapping3d */
} /* namespace cv */

#endif
