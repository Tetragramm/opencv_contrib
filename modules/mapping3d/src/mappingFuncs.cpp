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

#include "mappingFuncs.hpp"

//#ifdef TEST_EIGEN
//Eigen::Array2d getAzEl(const Eigen::Array3d& from, const Eigen::Array3d& to)
//{
//    Eigen::Array2d ret;
//    ret(0) = atan2((to(1) - from(1)), (to(0) - from(0)));
//    ret(1) = asin((to(2) - from(2)) / sqrt((to - from).square().sum()));
//    return ret;
//}
//#else
cv::Point2d getAzEl(const cv::Mat& from, const cv::Mat& to)
{
    cv::Point2d ret;
    cv::Point3d diff;
    diff.x = (to.at<double>(0) - from.at<double>(0));
    diff.y = (to.at<double>(1) - from.at<double>(1));
    diff.z = (to.at<double>(2) - from.at<double>(2));
    ret.x = atan2(diff.y, diff.x);
    ret.y = asin(diff.z / norm(diff));
    return ret;
}
//#endif