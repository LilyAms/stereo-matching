# stereo-matching
Stereo Matching with loopy belief propagation. Assignment from the computer vision course of ENPC. 

*Objective*
---
Given a left-right stereo image pair, the purpose of the assignment is to compute the left-right shift betwen the two images. 
This is done through the computation of a disparity map.
![] Materiel/imL.png
![] Materiel/imR.png

*Implementation*
---
Given two matching images, a disparity map is computed through the min-sum loopy belief propagation algorithm.
This is done to estimate the MAP (Maximum a posteriori)
