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
#include<time.h>
//using namespace cv;c语言没有命名空间？

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

int display = 1;



int main(int argc, char** argv)
{
	//计时
	clock_t start, finish;
	double totaltime;
	start = clock();
	IplImage* img1, *img2, *stacked;
	struct feature* feat1, *feat2, *feat, *feat3;
	struct feature** nbrs;
	//struct feature** import_feat1=NULL, **import_feat2=NULL;

	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, n3, k, i, m = 0;

	if (argc != 3)
		fatal_error("usage: %s <img1> <img2>", argv[0]);
	//加载图像
	img1 = cvLoadImage(argv[1], 1);
	if (!img1)
		fatal_error("unable to load image from %s", argv[1]);
	img2 = cvLoadImage(argv[2], 1);
	if (!img2)
		fatal_error("unable to load image from %s", argv[2]);
	stacked = stack_imgs(img1, img2);//得到宽度取二者最大，高度取二者高度之和的拼接图像，img1在上面
	//检测特征点
	n1 = 7875; n2 = 7333;
	//fprintf( stderr, "Finding features in %s...\n", argv[1] );
	//IplImage* down1 = cvCreateImage(cvSize(img1->width / 2, img1->height / 2), img1->depth, img1->nChannels);
	//cvPyrDown(img1, down1, 7);//filter=7 目前只支持CV_GAUSSIAN_5x5
	//n1 = sift_features( img1, &feat1 );
	//n3 = sift_features(down1, &feat3);//下采样的检测特征点
	//fprintf( stderr, "Finding features in %s...\n", argv[2] );
	//n2 = sift_features( img2, &feat2 );
	//借用siftfeat中的几句画特征点
	//if (display)
	//{

	//draw_features(img1, feat1, n1);
	// draw_features(img2, feat2, n2);
	// display_big_img(img1, argv[1]);
	// display_big_img(img2, argv[2]);
	// cvShowImage("downsample", down1);
	//
	// fprintf(stderr, "Found %d features in img1.\n", n1);
	// fprintf(stderr, "Found %d features in img2.\n", n2);
	// //cvWaitKey(0);
	//}
	char path1[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\boat1.txt";
	char path2[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\boat2.txt";
	import_features(path1, 1, &feat1);//第三个参数是指针的指针
	import_features(path2, 1, &feat2);
	finish = clock();
	/*feat1 = *import_feat1;
	feat2 = *import_feat2;
*/
  fprintf( stderr, "Building kd tree...\n" );
  kd_root = kdtree_build( feat2, n2 );//只对图2构造kd树
  for( i = 0; i < n1; i++ )//对图1的特征点遍历，在图2的kd树中找knn
    {
      feat = feat1 + i;//指针的加法
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );//近邻数是2
      if( k == 2 )//返回了两个nbars
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );//nbr是对每一个feat1中的特征点在kd树中找到的距离前2的点
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )//阈值为0.49
	    {
	      pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//最近邻小于次近邻的0.49时，把最近邻作为对应点
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
  //cvWaitKey( 0 );
 

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  
  {
    CvMat* H1,*H2;
    IplImage* xformed2,* xformed1;
	//double xpsnr;
	CvScalar scalar1,scalar2;
	//重新加载一遍，变换不希望有特征点箭头影响信噪比
	img1 = cvLoadImage(argv[1], 1);
	img2 = cvLoadImage(argv[2], 1);
	if (!img1)
		fatal_error("unable to load image from %s", argv[1]);
	img2 = cvLoadImage(argv[2], 1);
	if (!img2)
		fatal_error("unable to load image from %s", argv[2]);


    H1 = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//求变换矩阵用4对匹配对
		      homog_xfer_err, 3.0, NULL, NULL );//允许错误概率为0.01
	H2 = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//求变换矩阵用4对匹配对
		homog_xfer_err, 3.0, NULL, NULL);//允许错误概率为0.01
    if( H1 )
      {
		//打印矩阵
		printf("\n透视变换矩阵H为：");
		for (int i = 0; i < H1->rows; i++)//行  
		{
			for (int j = 0; j < H1->cols; j++)
			{
				//double H
				if (j % 3 == 0) printf("\n");
				printf("%9.3f", (float)cvGetReal2D(H1, i, j));
				//printf(" %f\t,H1->data.fl[i]");
			}
		}
	xformed1 = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );//通道数为3
	cvWarpPerspective( img1, xformed1, H1, 
			   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//线性插值
			   cvScalarAll( 255 ) );//255将没有填充部分填充为白色
	const char* path;
	path = "E:\\Right.png";
	cvSaveImage(path, xformed1, 0);
	//xformed2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);//通道数为3
	//cvWarpPerspective(img2, xformed2, H2,
	//	CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//线性插值
	//	cvScalarAll(0));//255将没有填充部分填充为白色
	//xformed1 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);//通道数为3
	//cvCopy(xformed, xformed1,NULL);
	//有一部分图像没有赋值，是黑色的
	for (int i = 0; i<xformed1->height; i++)
	{
		for (int j = 0; j<xformed1->width; j++)
		{
			scalar1 = cvGet2D(xformed1, i, j);
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
	
	psnr(img2, xformed1);
	
	cvNamedWindow( "Xformed1", 1 );
	cvShowImage( "Xformed1", xformed1);
	//cvShowImage("Xformed2", xformed2);
	
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\n此程序的运行时间为%f", totaltime);
	printf("\n");
	cvWaitKey( 0 );
	cvReleaseImage( &xformed1 );
	cvReleaseMat( &H1 );
	/*cvReleaseImage(&xformed2);
	cvReleaseMat(&H2);*/
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
		printf("\n\n两幅图像之间峰值信噪比为：%f\n", psnr);
	}
}