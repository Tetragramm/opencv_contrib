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
 // Copyright (C) 2016, OpenCV Foundation, all rights reserved.
 // Copyright (C) 2016, Jon Hoffman, all rights reserved.
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

#ifndef __OPENCV_MAPPING3D_HPP__
#define __OPENCV_MAPPING3D_HPP__

#include "opencv2/mapping3d/mapping3dBaseClasses.hpp"

/** @defgroup mapping3D Mapping 3D

Localizing a point in 3D space is a critical step for understanding the world in many applications.
These classes and functions are intended to provide a simple, efficient interface to calculate 3D states from many 2D measurements.

Classes
These classes are intended to save computational time by saving work and results, allowing updating results with each update efficiently.
StateCalculator (SCalc): This class provides a common interface to PCalc and PVCalc, allowing the use of whichever is appropriate without altering the interface.
PositionCalculator (PCalc): This class allows the combination of measurements from one or more cameras to calculate the position of a point.
PositionVelocityCalculator (PVCalc): This class allows the combination of measurements from one or more cameras to calculate the position and velocity of a point.

Functions
These functions are simple, one use instantiations of the classes.
calcPosition: This function allows the combination of measurements from one or more cameras to calculate the position of a point.
calcPositionVelocity: This function allows the combination of measurements from one or more cameras to calculate the position and velocity of a point.

*/

namespace cv
{
    namespace mapping3d
    {

        //! @addtogroup mapping3d
        //! @{

        /** @brief Calculates the 3D position of a point

        This function calculates the combination of measurements from one or more cameras to into the 3D position of a point based upon the paper 3DTMA.

        @param _tvecs Vector of camera translation vectors, ie. from estimatePose.
        @param _rvecs Vector of camera rotation angles, ie. from estimatePose.  Same length as _tvecs.
        @param _pts Vector of 2D object locations within the images, Point2f format, Same length as _tvecs.
        @param _sizes One Size or Vector of image sizes, in pixels
        @param _cameraMatrices One Mat or Vector of camera matrices, ie. from calibrateCamera
        @param _distortionMatrices One Mat or Vector of distortion matrices, ie. from calibrateCamera
        @param _location the 3D location of the object, relative to (0,0,0), a Mat.

        @sa  calibrateCamera PositionCalculator
        */
        CV_EXPORTS_W void calcPosition(InputArray _tvecs, InputArray _rvecs, InputArray _pts, Size _size,
            InputArray _cameraMatrices, InputArray _distortionMatrices, OutputArray _state);
        CV_EXPORTS_W void calcPosition(InputArray _tvecs, InputArray _rvecs, InputArray _pts, InputArray _sizes,
            InputArray _cameraMatrices, InputArray _distortionMatrices, OutputArray _state);

        //! @}

        //! @addtogroup mapping3d
        //! @{

        /** @brief Calculates the 3D position and velocity of a point

        This function calculates the combination of measurements from one or more cameras to into the 3D position and velocity of a point based upon the paper 3DTMA.

        @param _tvecs Vector of camera translation vectors, ie. from estimatePose.
        @param _rvecs Vector of camera rotation angles, ie. from estimatePose.  Same length as _tvecs.
        @param _pts Vector of 2D object locations within the images, Point2f format, Same length as _tvecs.
        @param _times Vector of measurement times, doubles, Same length ast _tvecs.
        @param _sizes One Size or Vector of image sizes, in pixels
        @param _cameraMatrices One Mat or Vector of camera matrices, ie. from calibrateCamera
        @param _distortionMatrices One Mat or Vector of distortion matrices, ie. from calibrateCamera
        @param _location the 3D location of the object, relative to (0,0,0), a Mat.

        @sa  calibrateCamera PositionVelocityCalculator
        */
        CV_EXPORTS_W void calcPositionVelocity(InputArray _tvecs, InputArray _rvecs, InputArray _pts, InputArray _times, 
            Size _size, InputArray _cameraMatrices, InputArray _distortionMatrices, double calcTime, OutputArray _state);
        CV_EXPORTS_W void calcPositionVelocity(InputArray _tvecs, InputArray _rvecs, InputArray _pts, InputArray _times,
            InputArray _sizes, InputArray _cameraMatrices, InputArray _distortionMatrices, double calcTime, OutputArray _state);

        //! @}

    }
}

#endif //__OPENCV_MAPPING3D_HPP__
