/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> //opencv2中是c++的接口
//#include <cxcore.h>
//#include <cvaux.h>

#include <math.h> 
#include <getpsnr.h>
//using namespace cv;c语言没有命名空间？

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

int display = 1;



int main( int argc, char** argv )
{
  IplImage* img1, * img2, * stacked;
  struct feature* feat1, * feat2, * feat;
  struct feature** nbrs;
  struct kd_node* kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int n1, n2, k, i, m = 0;
  
  if( argc != 3 )
    fatal_error( "usage: %s <img1> <img2>", argv[0] );

  img1 = cvLoadImage( argv[1], 1 );
  if( ! img1 )
    fatal_error( "unable to load image from %s", argv[1] );
  img2 = cvLoadImage( argv[2], 1 );
  if( ! img2 )
    fatal_error( "unable to load image from %s", argv[2] );
  stacked = stack_imgs( img1, img2 );

  fprintf( stderr, "Finding features in %s...\n", argv[1] );
  n1 = sift_features( img1, &feat1 );
  fprintf( stderr, "Finding features in %s...\n", argv[2] );
  n2 = sift_features( img2, &feat2 );
  //借用siftfeat中的几句画特征点
  if (display)
  {
	
	  draw_features(img1, feat1, n1);
	  draw_features(img2, feat2, n2);
	  display_big_img(img1, argv[1]);
	  display_big_img(img2, argv[2]);
	 
	  fprintf(stderr, "Found %d features in img1.\n", n1);
	  fprintf(stderr, "Found %d features in img2.\n", n2);
	  cvWaitKey(0);
  }
  fprintf( stderr, "Building kd tree...\n" );
  kd_root = kdtree_build( feat2, n2 );//只对图2构造kd树
  for( i = 0; i < n1; i++ )//对图1的特征点遍历，在图2的kd树中找knn
    {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
      if( k == 2 )
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
	    {
	      pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
	      pt2.y += img1->height;
	      cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	      m++;
	      feat1[i].fwd_match = nbrs[0];
	    }
	}
      free( nbrs );
    }

  fprintf( stderr, "Found %d total matches\n", m );
  display_big_img( stacked, "Matches" );
  cvWaitKey( 0 );
 

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  
  {
    CvMat* H;
    IplImage* xformed,* xformed1;
	//double xpsnr;
	CvScalar scalar1,scalar2;
    H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//求变换矩阵用4对匹配对
		      homog_xfer_err, 3.0, NULL, NULL );//允许错误概率为0.01
    if( H )
      {
	xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );//通道数为3
	cvWarpPerspective( img1, xformed, H, 
			   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//线性插值
			   cvScalarAll( 0 ) );//255将没有填充部分填充为白色
	//xformed1 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);//通道数为3
	//cvCopy(xformed, xformed1,NULL);
	//有一部分图像没有赋值，是黑色的
	for (int i = 0; i<xformed->height; i++)
	{
		for (int j = 0; j<xformed->width; j++)
		{
			scalar1 = cvGet2D(xformed, i, j);
			scalar2 = cvGet2D(img2, i, j);
			if (scalar1.val[0] == 0)
			{
				for (int c = 0; c < 3; c++)
				{
					//xformed(i,j)
					scalar1.val[c] = scalar2.val[c];
				}
			}

		}
	}
	
	psnr(img2, xformed);
	
	cvNamedWindow( "Xformed", 1 );
	cvShowImage( "Xformed", xformed);
	cvWaitKey( 0 );
	cvReleaseImage( &xformed );
	cvReleaseMat( &H );
      }
  }
  

  cvReleaseImage( &stacked );
  cvReleaseImage( &img1 );
  cvReleaseImage( &img2 );
  kdtree_release( kd_root );
  free( feat1 );
  free( feat2 );
  return 0;
}
void psnr(IplImage * src, IplImage * dst)
{
	IplImage * src_gray = cvCreateImage(cvGetSize(src), src->depth, 1);
	IplImage * dst_gray = cvCreateImage(cvGetSize(src), src->depth, 1);
	cvCvtColor(src, src_gray, CV_RGB2GRAY);
	cvCvtColor(dst, dst_gray, CV_RGB2GRAY);
	IplImage * img_gray = cvCreateImage(cvGetSize(src_gray), src_gray->depth, 1);
	cvAbsDiff(src_gray, dst_gray, img_gray);
	CvScalar scalar;
	double sum = 0;
	for (int i = 0; i<img_gray->height; i++)
	{
		for (int j = 0; j<img_gray->width; j++)
		{
			scalar = cvGet2D(img_gray, i, j);
			sum += scalar.val[0] * scalar.val[0];//获取的就是每一个像素点的灰度值
	
			//代表src图像BGR中的B通道的值
		}
	}
	double mse = 0;
	mse = sum / (img_gray->width * img_gray->height);
	if (mse == 0)
	{
		printf("相似度100%");
		printf("\n");
	}
	else
	{
		double psnr = 0;
		psnr = 10 * log10(255*255 / mse);
		printf("两幅图像之间峰值信噪比为：%f\n", psnr);
	}
}