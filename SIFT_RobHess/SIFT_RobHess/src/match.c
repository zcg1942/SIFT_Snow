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
#include <opencv2/highgui/highgui.hpp> //opencv2ÖĞÊÇc++µÄ½Ó¿Ú
//#include <cxcore.h>
//#include <cvaux.h>

#include <math.h> 
#include <getpsnr.h>
#include<time.h>
<<<<<<< HEAD
=======
#include <stdio.h>     
#include <unistd.h>  
>>>>>>> parent of e2d225b... PCAé™ç»´æˆ64ç»´å¯¼å‡ºtxt
//using namespace cv;cÓïÑÔÃ»ÓĞÃüÃû¿Õ¼ä£¿

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

int display = 1;



int main( int argc, char** argv )
{
	//¼ÆÊ±
	clock_t start, finish;
	double totaltime;
	start = clock();
  IplImage* img1, * img2, * stacked;
  struct feature* feat1, * feat2, * feat,* feat3;
  struct feature** nbrs;
  struct kd_node* kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int n1, n2,n3, k, i, m = 0;
  
  if( argc != 3 )
    fatal_error( "usage: %s <img1> <img2>", argv[0] );
  //¼ÓÔØÍ¼Ïñ
  img1 = cvLoadImage( argv[1], 1 );
  if( ! img1 )
    fatal_error( "unable to load image from %s", argv[1] );
  img2 = cvLoadImage( argv[2], 1 );
  if( ! img2 )
    fatal_error( "unable to load image from %s", argv[2] );
  stacked = stack_imgs( img1, img2 );
<<<<<<< HEAD
  n1 = 116; n2 = 95;//ÌØÕ÷µãÊıÁ¿ 
  ////¼ì²âÌØÕ÷µã
  //fprintf( stderr, "Finding features in %s...\n", argv[1] );
  //IplImage* down1 = cvCreateImage(cvSize(img1->width / 2, img1->height / 2), img1->depth, img1->nChannels);
  //cvPyrDown(img1, down1, 7);//filter=7 Ä¿Ç°Ö»Ö§³ÖCV_GAUSSIAN_5x5
  //n1 = sift_features( img1, &feat1 );
  //n3 = sift_features(down1, &feat3);//ÏÂ²ÉÑùµÄ¼ì²âÌØÕ÷µã
  //fprintf( stderr, "Finding features in %s...\n", argv[2] );
  //n2 = sift_features( img2, &feat2 );
  ////½èÓÃsiftfeatÖĞµÄ¼¸¾ä»­ÌØÕ÷µã
  //if (display)
  //{
	
	 // draw_features(img1, feat1, n1);
	 // draw_features(img2, feat2, n2);
	 // display_big_img(img1, argv[1]);
	 // display_big_img(img2, argv[2]);
	 // cvShowImage("downsample", down1);
	 //
	 // fprintf(stderr, "Found %d features in img1.\n", n1);
	 // fprintf(stderr, "Found %d features in img2.\n", n2);
	 // //cvWaitKey(0);
  //}
  char savepath11[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\feat1pca.txt";
  char savepath22[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\feat2pca.txt";
  import_features(savepath11,1,&feat1);
  import_features(savepath22,1, &feat2);
=======
  //¼ì²âÌØÕ÷µã
  fprintf( stderr, "Finding features in %s...\n", argv[1] );
  IplImage* down1 = cvCreateImage(cvSize(img1->width / 2, img1->height / 2), img1->depth, img1->nChannels);
  cvPyrDown(img1, down1, 7);//filter=7 Ä¿Ç°Ö»Ö§³ÖCV_GAUSSIAN_5x5
  n1 = sift_features( img1, &feat1 );//µÚ¶ş¸ö²ÎÊıÊÇÖ¸ÕëµÄµØÖ·
  n3 = sift_features(down1, &feat3);//ÏÂ²ÉÑùµÄ¼ì²âÌØÕ÷µã
  fprintf( stderr, "Finding features in %s...\n", argv[2] );
  n2 = sift_features( img2, &feat2 );
  //½èÓÃsiftfeatÖĞµÄ¼¸¾ä»­ÌØÕ÷µã
  if (display)
  {
	
	  draw_features(img1, feat1, n1);
	  draw_features(img2, feat2, n2);
	  display_big_img(img1, argv[1]);
	  display_big_img(img2, argv[2]);
	  cvShowImage("downsample", down1);
	 
	  fprintf(stderr, "Found %d features in img1.\n", n1);
	  fprintf(stderr, "Found %d features in img2.\n", n2);
	  //cvWaitKey(0);
  }
  /*ÀûÓÃPCA¶ÔÃèÊö×Ó½øĞĞ½µÎ¬
  feat1,feat2ÊÇÖ¸Ïòstruct featureµÄÖ¸Õë £¬feature½á¹¹ÌåÖĞ°üº¬ÁËÒ»¸ö128Î¬µÄdoubleÊı×é
  µ«Ò»·ùÍ¼ÏñÉÏÇ§¸öÌØÕ÷µã£¬ËüÃÇµÄÃèÊö×Ó¶¼´æ·ÅÔÚÄÄÀï
  °Ñ128Î¬Êı¾İÄÃ³öÀ´½µÎ¬»¹ÊÇÖ±½Ó¶Ôfeature½á¹¹½µÎ¬?
  Ò»ÖÖË¼Â·ÊÇ¶ÔÃèÊö×Ó½µÎ¬£¬Ò»ÖÖÊÇ¶ÔÌØÕ÷µã½µÎ¬
  »Ø¸´u010890209£ºÔ´ÂëÀï¾ÍÓĞ½«ÌØÕ÷ÃèÊöµ¼³öµ½ÎÄ¼şµÄº¯Êı£ºexport_features
  ÏÈµ¼³ö£¬PCAºóÔÙµ¼Èë£¬ÒòÎªÖ®ºóµÄkdÊ÷µÄ¹¹½¨ÊÇ¶Ôfeat²Ù×÷µÄ
  */
  //char* savepath = new char[100];
  char savepath1[80]="E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\feat1.txt";
  char savepath2[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\feat2.txt";
  //cÓïÑÔÖĞ\ÊÇ×ªÒå×Ö·û£¬ËùÒÔ×îºÃÓÃ\\
  //_getcwd(savepath1, sizeof(savepath));//»ñÈ¡µ±Ç°Â·¾¶£¬½«Â·¾¶ÓÃfopen´ò¿ª£¬·µ»Øfile
  export_features(savepath1, feat1, n1);//±£´æµÄ¸ñÊ½È¡¾öÓÚ feat[0].type;
  export_features(savepath2, feat2, n2);//Ó¦¸Ã¾ÍÊÇ°´ÕÕlowe¸ñÊ½±£´æµÄ
  //µ¼³öºóÔÙPCA·ÖÎöÔÙµ¼Èë£¬¼ÆËãºÄÊ±Ö»Ğè¼ÆËã±È½ÏÆ¥ÅäÊ±¼ä£¬²»Ëãµ¼Èëµ¼³öµÄÊ±¼ä
  
  import_features(savepath1, 1, feat1);//1´ú±íLowe¸ñÊ½
  import_features(savepath2, 1, feat2);
  char savepath11[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\feat1pca.txt";
  char savepath22[80] = "E:\\Local Repositories\\SIFT_Snow\\SIFT_RobHess\\SIFT_RobHess\\feat2pca.txt";
  export_features(savepath11, feat1, n1);//±£´æµÄ¸ñÊ½È¡¾öÓÚ feat[0].type;
  export_features(savepath22, feat2, n2);//Ó¦¸Ã¾ÍÊÇ°´ÕÕlowe¸ñÊ½±£´æµÄ

  
>>>>>>> parent of e2d225b... PCAé™ç»´æˆ64ç»´å¯¼å‡ºtxt

  fprintf( stderr, "Building kd tree...\n" );
  kd_root = kdtree_build( feat2, n2 );//Ö»¶ÔÍ¼2¹¹ÔìkdÊ÷
  for( i = 0; i < n1; i++ )//¶ÔÍ¼1µÄÌØÕ÷µã±éÀú£¬ÔÚÍ¼2µÄkdÊ÷ÖĞÕÒknn
    {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );//½üÁÚÊıÊÇ2
      if( k == 2 )//·µ»ØÁËÁ½¸önbars
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )//ãĞÖµÎª0.49
	    {
	      pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//×î½üÁÚĞ¡ÓÚ´Î½üÁÚµÄ0.49Ê±£¬°Ñ×î½üÁÚ×÷Îª¶ÔÓ¦µã
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
	//ÖØĞÂ¼ÓÔØÒ»±é£¬±ä»»²»Ï£ÍûÓĞÌØÕ÷µã¼ıÍ·Ó°ÏìĞÅÔë±È
	img1 = cvLoadImage(argv[1], 1);
	img2 = cvLoadImage(argv[2], 1);
	if (!img1)
		fatal_error("unable to load image from %s", argv[1]);
	img2 = cvLoadImage(argv[2], 1);
	if (!img2)
		fatal_error("unable to load image from %s", argv[2]);


    H1 = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//Çó±ä»»¾ØÕóÓÃ4¶ÔÆ¥Åä¶Ô
		      homog_xfer_err, 3.0, NULL, NULL );//ÔÊĞí´íÎó¸ÅÂÊÎª0.01
	H2 = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//Çó±ä»»¾ØÕóÓÃ4¶ÔÆ¥Åä¶Ô
		homog_xfer_err, 3.0, NULL, NULL);//ÔÊĞí´íÎó¸ÅÂÊÎª0.01
    if( H1 )
      {
		//´òÓ¡¾ØÕó
		printf("\nÍ¸ÊÓ±ä»»¾ØÕóHÎª£º");
		for (int i = 0; i < H1->rows; i++)//ĞĞ  
		{
			for (int j = 0; j < H1->cols; j++)
			{
				//double H
				if (j % 3 == 0) printf("\n");
				printf("%9.3f", (float)cvGetReal2D(H1, i, j));
				//printf(" %f\t,H1->data.fl[i]");
			}
		}
	xformed1 = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );//Í¨µÀÊıÎª3
	cvWarpPerspective( img1, xformed1, H1, 
			   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//ÏßĞÔ²åÖµ
			   cvScalarAll( 255 ) );//255½«Ã»ÓĞÌî³ä²¿·ÖÌî³äÎª°×É«
	const char* path;
	path = "E:\\Right.png";
	cvSaveImage(path, xformed1, 0);
	//xformed2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);//Í¨µÀÊıÎª3
	//cvWarpPerspective(img2, xformed2, H2,
	//	CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//ÏßĞÔ²åÖµ
	//	cvScalarAll(0));//255½«Ã»ÓĞÌî³ä²¿·ÖÌî³äÎª°×É«
	//xformed1 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);//Í¨µÀÊıÎª3
	//cvCopy(xformed, xformed1,NULL);
	//ÓĞÒ»²¿·ÖÍ¼ÏñÃ»ÓĞ¸³Öµ£¬ÊÇºÚÉ«µÄ
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
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\n´Ë³ÌĞòµÄÔËĞĞÊ±¼äÎª%f", totaltime);
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
			sum += scalar.val[0] * scalar.val[0];//»ñÈ¡µÄ¾ÍÊÇÃ¿Ò»¸öÏñËØµãµÄ»Ò¶ÈÖµ
	
			//´ú±ísrcÍ¼ÏñBGRÖĞµÄBÍ¨µÀµÄÖµ
		}
	}
	double mse = 0;
	mse = sum / (img_gray->width * img_gray->height);
	if (mse == 0)
	{
		printf("ÏàËÆ¶È100%");
		printf("\n");
	}
	else
	{
		double psnr = 0;
		psnr = 10 * log10(255*255 / mse);
		printf("\n\nÁ½·ùÍ¼ÏñÖ®¼ä·åÖµĞÅÔë±ÈÎª£º%f\n", psnr);
	}
}