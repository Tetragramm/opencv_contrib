Mapping in 3D, helping understand the 3D world from 2D sensors
============================================================

The purpose of this module is to provide a simple interface allowing the efficient calculation of 3D world points from multiple 2D measurements, along with other useful utilities.

The calcPosition and calcPositionVelocity functions and classes allow estimating the 3D position (and velocity) of a point based on the 2D measurements and the camera(s) intrinsic and extrinsic parameters. The method used is the Instrumental Variable TMA algorithm from [3DTMA].  Either one or many cameras can be used, at one or more times.  Multiple times are necessary for the estimation of velocity.

In addition, the calcObjectPosition and register3DPoints functions make it easier to align observations to a model.