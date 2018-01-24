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
#include <opencv2/highgui/highgui.hpp> //opencv2����c++�Ľӿ�
//#include <cxcore.h>
//#include <cvaux.h>

#include <math.h> 
#include <getpsnr.h>
#include<time.h>
void getRMSE(struct feature *feat1, struct feature *feat2, double Ha[]);
//#include<iostream>
//using namespace cv;c����û�������ռ䣿

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

int display = 1;

int n1;

int main( int argc, char** argv )
{
	//��ʱ
	clock_t start1, finish4;
	double totaltime;
	start1 = clock();

  IplImage* img1, * img2, * stacked;
  struct feature* feat1, *feat2, *feat, *feat3;
  struct feature** nbrs;
  struct kd_node* kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int n2,n3, k, i, m = 0;
  
  if( argc != 3 )
    fatal_error( "usage: %s <img1> <img2>", argv[0] );
  //����ͼ��
  img1 = cvLoadImage( argv[1], 1 );
  if( ! img1 )
    fatal_error( "unable to load image from %s", argv[1] );
  img2 = cvLoadImage( argv[2], 1 );
  if( ! img2 )
    fatal_error( "unable to load image from %s", argv[2] );
  cvShowImage("img2", img2);

  //���������
  fprintf( stderr, "Finding features in %s...\n", argv[1] );
  IplImage* down1 = cvCreateImage(cvSize(img1->width / 2, img1->height / 2), img1->depth, img1->nChannels);
  cvResize(img1, down1, CV_INTER_AREA);//�������Բ�ֵ������ڵõ���psnr��
  IplImage* down2 = cvCreateImage(cvSize(img2->width / 2, img2->height / 2), img2->depth, img2->nChannels);
  cvResize(img2, down2, CV_INTER_AREA);//�������Բ�ֵ������ڵõ���psnr��
  //����ڲ岹����ԭͼ������ֵ���㣬�Ը��������������

  //CvPoint2D32f srcTri[4], dstTri[4]; //��ά�����µĵ㣬����Ϊ����  
  //CvMat* warp_mat = cvCreateMat(3, 3, CV_32FC1);
  //srcTri[0].x = 0;
  //srcTri[0].y = 0;
  //srcTri[1].x = img1->width - 1;  //��Сһ������  
  //srcTri[1].y = 0;
  //srcTri[2].x = 0;
  //srcTri[2].y = img1->height - 1;
  //srcTri[3].x = img1->width - 1;  //bot right  
  //srcTri[3].y = img1->height - 1;

  //dstTri[0].x = 0;
  //dstTri[0].y = 0;
  //dstTri[1].x = down1->width - 1;
  //dstTri[1].y = 0;
  //dstTri[2].x = 0;
  //dstTri[2].y = down1->height- 1;
  //dstTri[3].x = down1->width - 1;
  //dstTri[3].y = down1->height - 1;

  
  //
  //cvPyrDown(img1, down1, 7);//filter=7 Ŀǰֻ֧��CV_GAUSSIAN_5x5
  stacked = stack_imgs(down1, down2);
  n1 = sift_features( down1, &feat1 );
  //n3 = sift_features(down1, &feat3);//�²����ļ��������
  fprintf( stderr, "Finding features in %s...\n", argv[2] );
  n2 = sift_features( down2, &feat2 );
  //����siftfeat�еļ��仭������
  if (display)
  {
	
	  draw_features(down1, feat1, n1);
	  draw_features(down2, feat2, n2);
	  display_big_img(down1, argv[1]);
	  display_big_img(down2, argv[2]);
	  cvShowImage("img1", img1);
	 
	  fprintf(stderr, "Found %d features in down1.\n", n1);
	  fprintf(stderr, "Found %d features in img2.\n", n2);
	  //cvWaitKey(0);
	 
  }
  
  fprintf( stderr, "Building kd tree...\n" );
  kd_root = kdtree_build( feat2, n2 );//ֻ��ͼ2����kd��
  for( i = 0; i < n1; i++ )//��ͼ1���������������ͼ2��kd������knn
    {
      feat = feat1 + i;//feat����ô�ܼ�int�ͣ�
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
      if( k == 2 )//��ά
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
	    {
	      pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
	      pt2.y += down1->height;//����ԭ�������Ͻǣ�����yӦ�ü�down1�ĸ߶Ƚ�������
	      cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	      m++;
	      feat1[i].fwd_match = nbrs[0];
	    }
	}
      free( nbrs );
    }

  fprintf( stderr, "Found %d total matches\n", m );
  display_big_img( stacked, "Matches" );
 // cvWaitKey( 0 );
  
 

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  
  {
    CvMat* H1;
    IplImage* xformed2,* xformed1;
	//double xpsnr;
	CvScalar scalar1,scalar2;
	int a = 0,b=0;
	//���¼���һ�飬�任��ϣ�����������ͷӰ�������
	img1 = cvLoadImage(argv[1], 1);
	if (!img1)
		fatal_error("unable to load image from %s", argv[1]);
	img2 = cvLoadImage(argv[2], 1);
	if (!img2)
		fatal_error("unable to load image from %s", argv[2]);


    H1 = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//��任������4��ƥ���
		      homog_xfer_err, 3.0, NULL, NULL );//����������Ϊ0.01
	//H2 = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//��任������4��ƥ���
		//homog_xfer_err, 3.0, NULL, NULL);//����������Ϊ0.01
    if( H1 )
      {
		double Ha[9] = { 0 };
		double Hb[9] = { 0 };
		printf("\n͸�ӱ任����HΪ��");
		for (int i = 0; i < H1->rows; i++)//��  
		{
			for (int j = 0; j < H1->cols; j++)
			{
				//double H
				if (b % 3 == 0) printf("\n");
				
				//Ha[(i + 1)*(j + 1)-1] = (double)cvGetReal2D(H1, i, j);
				//printf("%f\t", cvmGet(H1, i, j));
				Ha[a] = (double)cvGetReal2D(H1, i, j);//cvmat�о���ʹ�ӡ������������ת�ù�ϵ
				Hb[b] = (double)cvGetReal2D(H1, j, i);
 				//printf("%9.3f", Ha[(i + 1)*(j + 1)-1]);
				printf("%9.3f", Hb[b]);
				a++;
				b++;
				
				//printf(" %f\t,H1->data.fl[i]");
			}
			printf("\n");
		}
	xformed1 = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );//ͨ����Ϊ3
	cvResize(img1, down1, CV_INTER_LINEAR);//�������Բ�ֵ������ڵõ���psnr��
	Ha[6] = Ha[6] /2;
	Ha[7] = Ha[7] / 2;
	Ha[2] = Ha[2] * 2;
	Ha[5] = Ha[5] * 2;
	/*Ha[2] = Ha[2] / 2;
	Ha[5] = Ha[5] / 2;
	Ha[6] = Ha[6] * 2;
	Ha[7] = Ha[7] * 2;*/
	CvMat H2 = cvMat(3, 3, CV_64FC1, Ha);
	CvMat *H3 = &H2;
	//if (H3)
	//{
	//	double Hb[9] = { 0 };
	//	printf("\n͸�ӱ任����HΪ��");
	//	for (int i = 0; i < H3->rows; i++)//��  
	//	{
	//		for (int j = 0; j < H3->cols; j++)
	//		{
	//			//double H
	//			if (j % 3 == 0) printf("\n");

	//			//Ha[(i + 1)*(j + 1)-1] = (double)cvGetReal2D(H1, i, j);
	//			Ha[a] = (double)cvGetReal2D(H3, i, j);
	//			//printf("%9.3f", Ha[(i + 1)*(j + 1)-1]);
	//			printf("%9.3f", Ha[b]);
	//			b++;
	//			//printf(" %f\t,H1->data.fl[i]");
	//		}
	//	}
	
	cvWarpPerspective(img1, xformed1, H3,
		CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//���Բ�ֵ
		cvScalarAll(255));//255��û����䲿�����Ϊ��ɫ
	const char* path;
	path = "E:\\BothAREA.png";
		cvSaveImage( path,xformed1,0);
	//xformed2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);//ͨ����Ϊ3
	//cvWarpPerspective(img2, xformed2, H2,
	//	CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//���Բ�ֵ
	//	cvScalarAll(0));//255��û����䲿�����Ϊ��ɫ
	//xformed1 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);//ͨ����Ϊ3
	//cvCopy(xformed, xformed1,NULL);
	//��һ����ͼ��û�и�ֵ���Ǻ�ɫ��
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
	//getRMSE(feat1, feat2, Ha);
	
	cvNamedWindow( "Xformed1", 1 );
	cvShowImage( "Xformed1", xformed1);
	//cvShowImage("Xformed2", xformed2);
	finish4 = clock();
	totaltime = (double)(finish4 - start1) / CLOCKS_PER_SEC;
	printf("\n�˳��������ʱ��Ϊ%f", totaltime);
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
			sum += scalar.val[0] * scalar.val[0];//��ȡ�ľ���ÿһ�����ص�ĻҶ�ֵ
	
			//����srcͼ��BGR�е�Bͨ����ֵ
		}
	}
	double mse = 0;
	mse = sum / (img_gray->width * img_gray->height);
	if (mse == 0)
	{
		printf("���ƶ�100%");
		printf("\n");
	}
	else
	{
		double psnr = 0;
		psnr = 10 * log10(255*255 / mse);
		printf("\n����ͼ��֮���ֵ�����Ϊ��%f\n", psnr);
	}
}
//void getRMSE(struct feature *feat1, struct feature *feat2, double Ha[])
//{
//	
//
//	struct feature *feata, *featb;
//	double u, v, s, RMSE;
//	for (int i = 0; i < n1; i++)
//	{
//		feata = feat1 + i;//feat����ô�ܼ�int�ͣ�
//		featb = feat2 + i;
//		//IplImage*feati = cvCreateImage(cvSize(cvRound(feata->x), cvRound(feata->x)), IPL_DEPTH_8U, 3);//ͨ����Ϊ3
//		
//
//		u = ((feata->x)*Ha[0] + Ha[3] * (feata->y) + Ha[6]) / ((feata->x)*Ha[2] + Ha[5] * (feata->y) + Ha[8]);
//		v = ((featb->x)*Ha[1] + Ha[4] * (featb->y) + Ha[7]) / ((featb->x)*Ha[2] + Ha[5] * (featb->y) + Ha[8]);
//		double distance;
//		distance = powf((u - featb->x), 2) + powf((v - featb->y), 2);
//		//distance = sqrtf(distance);
//		s = 0;
//		s = s + distance;
//	}
//			 RMSE = sqrt(s / n1);
//			printf("\n����������ǣ�%f", RMSE);
//
//		
//	}
