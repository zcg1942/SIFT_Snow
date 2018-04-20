/*
  Functions for detecting SIFT image features.
  
  For more information, refer to:
  
  Lowe, D.  Distinctive image features from scale-invariant keypoints.
  <EM>International Journal of Computer Vision, 60</EM>, 2 (2004),
  pp.91--110.
  
  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  Note: The SIFT algorithm is patented in the United States and cannot be
  used in commercial products without a license from the University of
  British Columbia.  For more information, refer to the file LICENSE.ubc
  that accompanied this distribution.

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"

#include <cxcore.h>
#include <cv.h>

/************************* Local Function Prototypes *************************/

static IplImage* create_init_img( IplImage*, int, double );
static IplImage* convert_to_gray32( IplImage* );
static IplImage*** build_gauss_pyr( IplImage*, int, int, double );
static IplImage* downsample( IplImage* );
static IplImage*** build_dog_pyr( IplImage***, int, int );
static CvSeq* scale_space_extrema( IplImage***, int, int, double, int,
				   CvMemStorage*);
static int is_extremum( IplImage***, int, int, int, int );
static struct feature* interp_extremum( IplImage***, int, int, int, int, int,
					double);
static void interp_step( IplImage***, int, int, int, int, double*, double*,
			 double* );
static CvMat* deriv_3D( IplImage***, int, int, int, int );
static CvMat* hessian_3D( IplImage***, int, int, int, int );
static double interp_contr( IplImage***, int, int, int, int, double, double,
			    double );
static struct feature* new_feature( void );
static int is_too_edge_like( IplImage*, int, int, int );
static void calc_feature_scales( CvSeq*, double, int );
static void adjust_for_img_dbl( CvSeq* );
static void calc_feature_oris( CvSeq*, IplImage*** );
static double* ori_hist( IplImage*, int, int, int, int, double );
static int calc_grad_mag_ori( IplImage*, int, int, double*, double* );
static void smooth_ori_hist( double*, int );
static double dominant_ori( double*, int );
static void add_good_ori_features( CvSeq*, double*, int, double,
				   struct feature* );
static struct feature* clone_feature( struct feature* );
static void compute_descriptors( CvSeq*, IplImage***, int, int );
static double*** descr_hist( IplImage*, int, int, double, double, int, int );
static void interp_hist_entry( double***, double, double, double, double, int,
			       int);
static void hist_to_descr( double***, int, int, struct feature* );
static void normalize_descr( struct feature* );
static int feature_cmp( void*, void*, void* );
static void release_descr_hist( double****, int );
static void release_pyr( IplImage****, int, int );


/*********************** Functions prototyped in sift.h **********************/


/**
   Finds SIFT features in an image using default parameter values.  All
   detected features are stored in the array pointed to by \a feat.

   @param img the image in which to detect features
   @param feat a pointer to an array in which to store detected features

   @return Returns the number of features stored in \a feat or -1 on failure
   @see _sift_features()
*/
int sift_features( IplImage* img, struct feature** feat )
{
  return _sift_features( img, feat, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
			 SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
			 SIFT_DESCR_HIST_BINS );
}



/**
   Finds SIFT features in an image using user-specified parameter values.  All
   detected features are stored in the array pointed to by \a feat.

   @param img the image in which to detect features
   @param fea a pointer to an array in which to store detected features
   @param intvls the number of intervals sampled per octave of scale space
   @param sigma the amount of Gaussian smoothing applied to each image level
     before building the scale space representation for an octave
   @param cont_thr a threshold on the value of the scale space function
     \f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
     feature location and scale, used to reject unstable features;  assumes
     pixel values in the range [0, 1]
   @param curv_thr threshold on a feature's ratio of principle curvatures
     used to reject features that are too edge-like
   @param img_dbl should be 1 if image doubling prior to scale space
     construction is desired or 0 if not
   @param descr_width the width, \f$n\f$, of the \f$n \times n\f$ array of
     orientation histograms used to compute a feature's descriptor
   @param descr_hist_bins the number of orientations in each of the
     histograms in the array used to compute a feature's descriptor

   @return Returns the number of keypoints stored in \a feat or -1 on failure
   @see sift_keypoints()
*/
int _sift_features( IplImage* img, struct feature** feat, int intvls,
		    double sigma, double contr_thr, int curv_thr,
		    int img_dbl, int descr_width, int descr_hist_bins )
{
  IplImage* init_img;
  IplImage*** gauss_pyr, *** dog_pyr;
  CvMemStorage* storage;
  CvSeq* features;//注意到这里是复数，代表一幅图像所有的特征点
  int octvs, i, n = 0;
  
  /* check arguments */
  if( ! img )
    fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );
  if( ! feat )
    fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

  /* build scale space pyramid; smallest dimension of top level is ~4 pixels */
  /* 算法第一步，初始化图像 */
  init_img = create_init_img( img, img_dbl, sigma );
  octvs = log( MIN( init_img->width, init_img->height ) ) / log(2) - 2;//octvs是整个金字塔组数，用了换底公式
  gauss_pyr = build_gauss_pyr( init_img, octvs, intvls, sigma );//建立高斯金字塔
  dog_pyr = build_dog_pyr( gauss_pyr, octvs, intvls );//建立高斯差分金字塔，octvs是金字塔组数，intvls是层数（每组金字塔有几张图片）
  
  storage = cvCreateMemStorage( 0 );
  /* 算法第三步，寻找尺度空间极值，contr_thr是去除对比度低的点所采用的阀值，curv_thr是去除边缘特征所采取的阀值 */
  features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr,
	  curv_thr, storage);
	  /* 算法第四步，计算特征向量的尺度 */
	  calc_feature_scales(features, sigma, intvls);
  /* 算法第五步，调整图像的大小 */
  if( img_dbl )
    adjust_for_img_dbl( features );//转换为struct feature类型
  /* 算法第六步，计算特征点的主要方向 */
  calc_feature_oris( features, gauss_pyr );
  /* 算法第七步，计算描述子，其中包括计算二维方向直方图并转换直方图为特征描述子 */
  compute_descriptors( features, gauss_pyr, descr_width, descr_hist_bins );

  /* sort features by decreasing scale and move from CvSeq to array */
  /* 算法第八步，按尺度大小对描述子进行排序 */
  cvSeqSort( features, (CvCmpFunc)feature_cmp, NULL );
  n = features->total;
  *feat = calloc( n, sizeof(struct feature) );
  *feat = cvCvtSeqToArray( features, *feat, CV_WHOLE_SEQ );//将所有特征点分别提取出来成数列
  for( i = 0; i < n; i++ )
    {
      free( (*feat)[i].feature_data );
      (*feat)[i].feature_data = NULL;
    }
  
  cvReleaseMemStorage( &storage );
  cvReleaseImage( &init_img );
  release_pyr( &gauss_pyr, octvs, intvls + 3 );
  release_pyr( &dog_pyr, octvs, intvls + 2 );
  return n;
}


/************************ Functions prototyped here **************************/

/*
  Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
  optionally doubled in size prior to smoothing.

  @param img input image
  @param img_dbl if true, image is doubled in size prior to smoothing
  @param sigma total std of Gaussian smoothing
*/
static IplImage* create_init_img( IplImage* img, int img_dbl, double sigma )
{
  IplImage* gray, * dbl;
  double sig_diff;

  gray = convert_to_gray32( img );//转换图像为32位灰度图
  if( img_dbl )
    {
      sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );//????????
	  // Lowe 建议在建立尺度空间前首先对原始图像长宽扩展一倍，以保留原始图像信息，增加特征点数量
      dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ),//cvCreateImage创建图像首地址
			   IPL_DEPTH_32F, 1 );//IPL_DEPTH_32F - 单精度浮点数
      cvResize( gray, dbl, CV_INTER_CUBIC );//缩放，双三次插值
      cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );//高斯平滑，sig_diff为标准差
      cvReleaseImage( &gray );//IplImage*型的变量值赋为NULL，而这个变量本身还是存在的并且在内存中的存储位置不变
      return dbl;
    }
  else
    {
      sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );
      cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
      return gray;
    }
}



/*
  Converts an image to 32-bit grayscale

  @param img a 3-channel 8-bit color (BGR) or 8-bit gray image

  @return Returns a 32-bit grayscale image
*/
static IplImage* convert_to_gray32( IplImage* img )
{
  IplImage* gray8, * gray32;
  
  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  if( img->nChannels == 1 )
    gray8 = cvClone( img );
  else
    {
      gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );//图像像素的位深度,IPL_DEPTH_8U - 8位无符号整数
      cvCvtColor( img, gray8, CV_BGR2GRAY );//颜色空间转换函数,参数CV_BGR2GRAY是RGB到gray
    }
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );//使用线性变换转换数组,第三个参数是比例因子

  cvReleaseImage( &gray8 );//cvReleaseImage函数只是将IplImage*型的变量值赋为NULL
  return gray32;
}



/*
  Builds Gaussian scale space pyramid from an image
  建立高斯尺度空间
  @param base base image of the pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave
  @param sigma amount of Gaussian smoothing per octave

  @return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3)
    array
*/
static IplImage*** build_gauss_pyr( IplImage* base, int octvs,
			     int intvls, double sigma )
{
  IplImage*** gauss_pyr;//指针的指针的指针？？？？？？？三级指针
  //const int _intvls = intvls;
  //double sig[_intvls+3], sig_total, sig_prev, k;
  double *sig = (double *)calloc(intvls + 3, sizeof(double)); double sig_total, sig_prev, k;
  int i, o;
  //为高斯金字塔gauss_pyr分配空间，共octvs个元素，每个元素是一组图像的首指针
  gauss_pyr = calloc( octvs, sizeof( IplImage** ) );
  for( i = 0; i < octvs; i++ )
    gauss_pyr[i] = calloc( intvls + 3, sizeof( IplImage *) );

  /*
    precompute Gaussian sigmas using the following formula:

    \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2

    sig[i] is the incremental sigma value needed to compute 
    the actual sigma of level i. Keeping track of incremental
    sigmas vs. total sigmas keeps the gaussian kernel small.
  */
  k = pow( 2.0, 1.0 / intvls );//2的幂次方，求金字塔总组数
  sig[0] = sigma;
  sig[1] = sigma * sqrt( k*k- 1 );//？？？？？？？
  for (i = 2; i < intvls + 3; i++)
      sig[i] = sig[i-1] * k;//同一个组相邻层之间尺度参数差k倍

  for( o = 0; o < octvs; o++ )
    for( i = 0; i < intvls + 3; i++ )//为了连续，相邻层间取极值点，所以要加3
      {
	if( o == 0  &&  i == 0 )//第一组第一层
	  gauss_pyr[o][i] = cvCloneImage(base);

	/* base of new octvave is halved image from end of previous octave */
	else if( i == 0 )//其他组第一层
	  gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intvls] );//前一租倒数第三层的下采样
	  
	/* blur the current octave's last image to create the next one */
	else
	  {
	    gauss_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i-1]),
					     IPL_DEPTH_32F, 1 );
	    cvSmooth( gauss_pyr[o][i-1], gauss_pyr[o][i],
		      CV_GAUSSIAN, 0, 0, sig[i], sig[i] );
	  }
      }
  free(sig);
  return gauss_pyr;
}



/*
  Downsamples an image to a quarter of its size (half in each dimension)
  using nearest-neighbor interpolation

  @param img an image

  @return Returns an image whose dimensions are half those of img
*/
static IplImage* downsample( IplImage* img )
{
  IplImage* smaller = cvCreateImage( cvSize(img->width / 2, img->height / 2),
				     img->depth, img->nChannels );
  cvResize( img, smaller, CV_INTER_NN );//最近邻插补，从原图的像素值计算，对浮点坐标找最近的像素值
  //？？？？为什么扩展原图时用双三次插值，下采样用最近邻会失真严重
  return smaller;
}



/*
  Builds a difference of Gaussians scale space pyramid by subtracting adjacent
  intervals of a Gaussian pyramid
  构建差分高斯金字塔，相邻层之间相减
  @param gauss_pyr Gaussian scale-space pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave

  @return Returns a difference of Gaussians scale space pyramid as an
    octvs x (intvls + 2) array
*/
static IplImage*** build_dog_pyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
  IplImage*** dog_pyr;
  int i, o;

  dog_pyr = calloc( octvs, sizeof( IplImage** ) );//动态分配内存，将所分配的内存空间中的每一位都初始化为零
  for( i = 0; i < octvs; i++ )
    dog_pyr[i] = calloc( intvls + 2, sizeof(IplImage*) );//DoG比高斯金字塔层数少一层，+3变为+2

  for( o = 0; o < octvs; o++ )
    for( i = 0; i < intvls + 2; i++ )
      {
	dog_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i]),
				       IPL_DEPTH_32F, 1 );
	cvSub( gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL );//矩阵做减法，元素级相减
      }

  return dog_pyr;
}



/*
  Detects features at extrema in DoG scale space.  Bad features are discarded
  based on contrast and ratio of principal curvatures.
  在DoG尺度空间上找极值

  @param dog_pyr DoG scale space pyramid
  @param octvs octaves of scale space represented by dog_pyr
  @param intvls intervals per octave
  @param contr_thr low threshold on feature contrast
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  @return Returns an array of detected features whose scales, orientations,
    and descriptors are yet to be determined.
*/
static CvSeq* scale_space_extrema( IplImage*** dog_pyr, int octvs, int intvls,
				   double contr_thr, int curv_thr,
				   CvMemStorage* storage )
{
  CvSeq* features;
  double prelim_contr_thr = 0.5 * contr_thr / intvls;
  //contr_thr是去除对比度低的点所采用的阈值 curv_thr是去除边缘特征的阈值
  struct feature* feat;//数组指针，用来存储图像的特征向量
  struct detection_data* ddata;
  int o, i, r, c;
  unsigned long* feature_mat;

  features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );
  for( o = 0; o < octvs; o++ )
  {
    feature_mat = calloc( dog_pyr[o][0]->height * dog_pyr[o][0]->width, sizeof(unsigned long) );
    for( i = 1; i <= intvls; i++ )
      for(r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SIFT_IMG_BORDER; r++)
	for(c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SIFT_IMG_BORDER; c++)
	  /* perform preliminary check on contrast */
	  /* 预判断对比度，如果这都过不了，该点对比度实在是非常低没必要再往下进行直接舍去 */
	  if( ABS( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
	    if( is_extremum( dog_pyr, o, i, r, c ) )//如果是极值点
	      {
		feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
		//返回值非空，表明此点已被成功修正
		if( feat )
		  {
		    ddata = feat_detection_data( feat );
		    if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],//去除边缘响应
					    ddata->r, ddata->c, curv_thr ) )
		      {
                        if( ddata->intvl > sizeof(unsigned long) )
                          cvSeqPush( features, feat );
                        else if( (feature_mat[dog_pyr[o][0]->width * ddata->r + ddata->c] & (1 << ddata->intvl-1)) == 0 )
                        {
                          cvSeqPush( features, feat );//压栈，序列在内部其实对应一个双端序列,很自然地将序列做一个栈使用
                          feature_mat[dog_pyr[o][0]->width * ddata->r + ddata->c] += 1 << ddata->intvl-1;
                        }
		      }
		    else
		      free( ddata );
		    free( feat );
		  }
	      }
    free( feature_mat );
  }
  return features;
}



/*
  Determines whether a pixel is a scale-space extremum by comparing it to it's
  3x3x3 pixel neighborhood.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's scale space octave
  @param intvl pixel's within-octave interval
  @param r pixel's image row
  @param c pixel's image col

  @return Returns 1 if the specified pixel is an extremum (max or min) among
    it's 3x3x3 pixel neighborhood.
	判断是否是极值点
*/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	double val = pixval32f(dog_pyr[octv][intvl], r, c); //调用函数pixval32f获取图像dog_pyr[octv][intvl]的第r行第c列的点的坐标值
  int i, j, k;

  /* check for maximum */
  if( val > 0 )
    {
      for( i = -1; i <= 1; i++ )
	for( j = -1; j <= 1; j++ )
	  for( k = -1; k <= 1; k++ )
	    if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
	      return 0;
    }

  /* check for minimum */
  else
    {
      for( i = -1; i <= 1; i++ )
	for( j = -1; j <= 1; j++ )
	  for( k = -1; k <= 1; k++ )
	    if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
	      return 0;
    }

  return 1;
}



/*
  Interpolates a scale-space extremum's location and scale to subpixel
  accuracy to form an image feature.  Rejects features with low contrast.
  Based on Section 4 of Lowe's paper.  

  获取亚像素的极值点的位置

  @param dog_pyr DoG scale space pyramid
  @param octv feature's octave of scale space
  @param intvl feature's within-octave interval
  @param r feature's image row
  @param c feature's image column
  @param intvls total intervals per octave
  @param contr_thr threshold on feature contrast

  @return Returns the feature resulting from interpolation of the given
    parameters or NULL if the given location could not be interpolated or
    if contrast at the interpolated loation was too low.  If a feature is
    returned, its scale, orientation, and descriptor are yet to be determined.
	修正
*/
static struct feature* interp_extremum( IplImage*** dog_pyr, int octv,
					int intvl, int r, int c, int intvls,
					double contr_thr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double xi, xr, xc, contr;
  int i = 0;//插值次数
  //SIFT_MAX_INTERP_STEPS指定了关键点的最大插值次数，即最多修正多少次，默认是5
  while( i < SIFT_MAX_INTERP_STEPS )
    {
      interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
	  //若在任意方向上的偏移量大于0.5时，意味着差值中心已经偏移到它的临近点上，所以必须改变当前关键点的位置坐标
      if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
		  //若三方向上偏移量都小于0.5，表示已经够精确，则不用继续插值
	break;
      
      c += cvRound( xc );
      r += cvRound( xr );
      intvl += cvRound( xi );//σ方向，即层方向

	  //若坐标修正后超出范围，则结束插值，返回NULL
      if( intvl < 1  ||
	  intvl > intvls  ||
	  c < SIFT_IMG_BORDER  ||
	  r < SIFT_IMG_BORDER  ||
	  c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER  ||
	  r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER )
	{
	  return NULL;
	}
      
      i++;
    }
  
  /* ensure convergence of interpolation */
  if( i >= SIFT_MAX_INTERP_STEPS )
    return NULL;
  //计算被插值点的对比度：D + 0.5 * dD^T * X
  contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
  if( ABS( contr ) < contr_thr / intvls )
    return NULL;

  feat = new_feature();
  ddata = feat_detection_data( feat );
  feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );
  feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
  ddata->r = r;
  ddata->c = c;
  ddata->octv = octv;
  ddata->intvl = intvl;
  ddata->subintvl = xi;

  return feat;
}



/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl interval being interpolated
  @param r row being interpolated
  @param c column being interpolated
  @param xi output as interpolated subpixel increment to interval
  @param xr output as interpolated subpixel increment to row
  @param xc output as interpolated subpixel increment to col
*/

static void interp_step( IplImage*** dog_pyr, int octv, int intvl, int r, int c,
			 double* xi, double* xr, double* xc )
{
  CvMat* dD, * H, * H_inv, X;
  double x[3] = { 0 };
  
  dD = deriv_3D( dog_pyr, octv, intvl, r, c );
  H = hessian_3D( dog_pyr, octv, intvl, r, c );
  H_inv = cvCreateMat( 3, 3, CV_64FC1 );
  cvInvert( H, H_inv, CV_SVD );
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );
  
  cvReleaseMat( &dD );
  cvReleaseMat( &H );
  cvReleaseMat( &H_inv );

  *xi = x[2];
  *xr = x[1];
  *xc = x[0];
}



/*
  Computes the partial derivatives in x, y, and scale of a pixel in the DoG
  scale space pyramid.
  计算三维偏导数

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the vector of partial derivatives for pixel I
    { dI/dx, dI/dy, dI/ds }^T as a CvMat*
*/
static CvMat* deriv_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
  CvMat* dI;
  double dx, dy, ds;

  dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
	 pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
  dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
	 pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
  ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
	 pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;
  
  dI = cvCreateMat( 3, 1, CV_64FC1 );
  cvmSet( dI, 0, 0, dx );
  cvmSet( dI, 1, 0, dy );
  cvmSet( dI, 2, 0, ds );

  return dI;
}



/*
  Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
  计算三维海森矩阵

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the Hessian matrix (below) for pixel I as a CvMat*

  / Ixx  Ixy  Ixs \ <BR>
  | Ixy  Iyy  Iys | <BR>
  \ Ixs  Iys  Iss /
*/
static CvMat* hessian_3D( IplImage*** dog_pyr, int octv, int intvl, int r,
			  int c )
{
  CvMat* H;
  double v, dxx, dyy, dss, dxy, dxs, dys;
  
  v = pixval32f( dog_pyr[octv][intvl], r, c );
  dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
	  pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
  dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
	  pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
  dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
	  pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );
  dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
	  pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
	  pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
	  pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
  dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
	  pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
	  pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
	  pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
  dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
	  pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
	  pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
	  pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;
  
  H = cvCreateMat( 3, 3, CV_64FC1 );
  cvmSet( H, 0, 0, dxx );
  cvmSet( H, 0, 1, dxy );
  cvmSet( H, 0, 2, dxs );
  cvmSet( H, 1, 0, dxy );
  cvmSet( H, 1, 1, dyy );
  cvmSet( H, 1, 2, dys );
  cvmSet( H, 2, 0, dxs );
  cvmSet( H, 2, 1, dys );
  cvmSet( H, 2, 2, dss );

  return H;
}



/*
  Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
  paper.
  计算插入像素的对比度

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl within-octave interval
  @param r pixel row
  @param c pixel column
  @param xi interpolated subpixel increment to interval
  @param xr interpolated subpixel increment to row
  @param xc interpolated subpixel increment to col

  @param Returns interpolated contrast.
*/
static double interp_contr( IplImage*** dog_pyr, int octv, int intvl, int r,
			    int c, double xi, double xr, double xc )
{
  CvMat* dD, X, T;
  double t[1], x[3] = { xc, xr, xi };

  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
  dD = deriv_3D( dog_pyr, octv, intvl, r, c );
  cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
  cvReleaseMat( &dD );

  return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}



/*
  Allocates and initializes a new feature

  @return Returns a pointer to the new feature
*/
static struct feature* new_feature( void )
{
  struct feature* feat;
  struct detection_data* ddata;

  feat = malloc( sizeof( struct feature ) );
  memset( feat, 0, sizeof( struct feature ) );
  ddata = malloc( sizeof( struct detection_data ) );
  memset( ddata, 0, sizeof( struct detection_data ) );
  feat->feature_data = ddata;
  feat->type = FEATURE_LOWE;

  return feat;
}



/*
  Determines whether a feature is too edge like to be stable by computing the
  ratio of principal curvatures at that feature.  Based on Section 4.1 of
  Lowe's paper.

  通过计算主曲率去除边缘响应

  @param dog_img image from the DoG pyramid in which feature was detected
  @param r feature row
  @param c feature col
  @param curv_thr high threshold on ratio of principal curvatures

  @return Returns 0 if the feature at (r,c) in dog_img is sufficiently
    corner-like or 1 otherwise.
*/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
  double d, dxx, dyy, dxy, tr, det;

  /*某点的主曲率与其海森矩阵的特征值成正比，为了避免直接计算特征值，这里只考虑特征值的比值
  可通过计算海森矩阵的迹tr(H)和行列式det(H)来计算特征值的比值
  设a是海森矩阵的较大特征值，b是较小的特征值，有a = r*b，r是大小特征值的比值
  tr(H) = a + b; det(H) = a*b;
  tr(H)^2 / det(H) = (a+b)^2 / ab = (r+1)^2/r
  r越大，越可能是边缘点；伴随r的增大，(r+1)^2/r 的值也增大，所以可通过(r+1)^2/r 判断主曲率比值是否满足条件*/

  /* principal curvatures are computed using the trace and det of Hessian */
  d = pixval32f(dog_img, r, c);
  //用差分近似代替偏导，求出海森矩阵的几个元素值

  dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
  dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
  dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
	  pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
  tr = dxx + dyy;
  det = dxx * dyy - dxy * dxy;//海森矩阵的行列式  

  /* negative determinant -> curvatures have different signs; reject feature */
  //若行列式为负，表明曲率有不同的符号，去除此点  
  if( det <= 0 )
    return 1;

  if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
    return 0;
  return 1;
}



/*
  Calculates characteristic scale for each feature in an array.
  计算特征向量的尺度

  @param features array of features
  @param sigma amount of Gaussian smoothing per octave of scale space
  @param intvls intervals per octave of scale space
*/
static void calc_feature_scales( CvSeq* features, double sigma, int intvls )
{
  struct feature* feat;
  struct detection_data* ddata;
  double intvl;
  int i, n;

  n = features->total;
  for( i = 0; i < n; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      ddata = feat_detection_data( feat );
      intvl = ddata->intvl + ddata->subintvl;//特征点所在的层数ddata->intvl加上特征点在层方向上的亚像素偏移量，得到特征点的较为精确的层数  
      feat->scl = sigma * pow( 2.0, ddata->octv + intvl / intvls );
	  //计算特征点所在的组的尺度，给detection_data的scl_octv成员赋值
      ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}



/*
  Halves feature coordinates and scale in case the input image was doubled
  prior to scale space construction.
  调整图像大小
  将特征点序列中每个特征点的坐标减半(当设置了将图像放大为原图的2倍时，特征点检测完之后调用)

  @param features array of features
*/
static void adjust_for_img_dbl( CvSeq* features )
{
  struct feature* feat;
  int i, n;

  n = features->total;
  for( i = 0; i < n; i++ )
    {
	  //调用宏，获取序列features中的第i个元素，并强制转换为struct feature类型  
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
	  //将特征点的x,y坐标和尺度都减半  
      feat->x /= 2.0;
      feat->y /= 2.0;
      feat->scl /= 2.0;
      feat->img_pt.x /= 2.0;
      feat->img_pt.y /= 2.0;
    }
}



/*
  Computes a canonical orientation for each image feature in an array.  Based
  on Section 5 of Lowe's paper.  This function adds features to the array when
  there is more than one dominant orientation at a given feature location.
  计算特征点主方向

  @param features an array of image features
  @param gauss_pyr Gaussian scale space pyramid
*/
static void calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double* hist;//存放梯度直方图的数组
  double omax;
  int i, j, n = features->total;

  for( i = 0; i < n; i++ )
    {
	  //给每个特征点分配feature结构大小的内存  
      feat = malloc( sizeof( struct feature ) );
	  //移除列首元素，放到feat中  
      cvSeqPopFront( features, feat );
      ddata = feat_detection_data( feat );

	  //计算指定像素点的梯度方向直方图，返回存放直方图的数组给hist
      hist = ori_hist( gauss_pyr[ddata->octv][ddata->intvl],  //特征点所在的图像 
		       ddata->r, ddata->c, //特征点的行列坐标
			   SIFT_ORI_HIST_BINS,          //默认的梯度直方图的bin(柱子)个数
			   cvRound(SIFT_ORI_RADIUS * ddata->scl_octv),   //特征点方向赋值过程中，搜索邻域的半径为：3 * 1.5 * σ  
		       SIFT_ORI_SIG_FCTR * ddata->scl_octv );//计算直翻图时梯度幅值的高斯权重的初始值  
	  //对梯度直方图进行高斯平滑，弥补因没有仿射不变性而产生的特征点不稳定的问题,SIFT_ORI_SMOOTH_PASSES指定了平滑次数
      for( j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
	smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );
	  //查找梯度直方图中主方向的梯度幅值，即查找直方图中最大bin的值,返回给omax  
      omax = dominant_ori( hist, SIFT_ORI_HIST_BINS );
      add_good_ori_features( features, hist, SIFT_ORI_HIST_BINS,
			     omax * SIFT_ORI_PEAK_RATIO, feat );
      free( ddata );
      free( feat );
      free( hist );
    }
}



/*
  Computes a gradient orientation histogram at a specified pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param n number of histogram bins
  @param rad radius of region over which histogram is computed
  @param sigma std for Gaussian weighting of histogram entries

  @return Returns an n-element array containing an orientation histogram
    representing orientations between 0 and 2 PI.
*/
static double* ori_hist( IplImage* img, int r, int c, int n, int rad,
			 double sigma )
{
  double* hist;
  double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
  int bin, i, j;

  //为直方图数组分配空间，共n个元素，n是柱的个数 
  hist = calloc( n, sizeof( double ) );
  exp_denom = 2.0 * sigma * sigma;
  for( i = -rad; i <= rad; i++ )
    for( j = -rad; j <= rad; j++ )
		//计算指定点的梯度的幅值mag和方向ori，返回值为1表示计算成功  
      if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
	{
	  w = exp( -( i*i + j*j ) / exp_denom );//该点的梯度幅值权重  
	  bin = cvRound( n * ( ori + CV_PI ) / PI2 );//计算梯度的方向对应的直方图中的bin下标  
	  bin = ( bin < n )? bin : 0;
	  hist[bin] += w * mag;//在直方图的某个bin中累加加权后的幅值 
	}

  return hist;
}



/*
  Calculates the gradient magnitude and orientation at a given pixel.
  计算指定点的梯度的幅值magnitude和方向orientation

  @param img image
  @param r pixel row
  @param c pixel col
  @param mag output as gradient magnitude at pixel (r,c)
  @param ori output as gradient orientation at pixel (r,c)

  @return Returns 1 if the specified pixel is a valid one and sets mag and
    ori accordingly; otherwise returns 0
*/
static int calc_grad_mag_ori( IplImage* img, int r, int c, double* mag,
			      double* ori )
{
  double dx, dy;

  if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 )
    {
      dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );//x方向偏导，r，c是行列  
      dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
      *mag = sqrt( dx*dx + dy*dy );//梯度的幅值，即梯度的模  
      *ori = atan2( dy, dx );//梯度的方向
      return 1;
    }

  else
    return 0;
}



/*
  Gaussian smooths an orientation histogram.
  对梯度方向直方图进行高斯平滑，弥补因没有仿射不变性而产生的特征点不稳定的问题
  参数：
  hist：存放梯度直方图的数组
  n：梯度直方图中bin的个数
  @param hist an orientation histogram
  @param n number of bins
*/
static void smooth_ori_hist( double* hist, int n )
{
  double prev, tmp, h0 = hist[0];
  int i;

  prev = hist[n-1];
  //类似均值漂移的一种邻域平滑，减少突变的影响 
  for( i = 0; i < n; i++ )
    {
      tmp = hist[i];
      hist[i] = 0.25 * prev + 0.5 * hist[i] + 
	0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
      prev = tmp;
    }
}



/*
  Finds the magnitude of the dominant orientation in a histogram

  @param hist an orientation histogram
  @param n number of bins

  @return Returns the value of the largest bin in hist
*/
static double dominant_ori( double* hist, int n )
{
  double omax;
  int maxbin, i;

  omax = hist[0];
  maxbin = 0;
  //遍历直方图，找到最大的bin  
  for( i = 1; i < n; i++ )
    if( hist[i] > omax )
      {
	omax = hist[i];
	maxbin = i;
      }
  return omax;
}



/*
  Interpolates a histogram peak from left, center, and right values
  根据左、中、右三个bin的值对当前bin进行直方图插值，以求取更精确的方向角度值
*/

#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )



/*
  Adds features to an array for every orientation in a histogram greater than
  a specified threshold.
  若当前特征点的直方图中某个bin的值大于给定的阈值，则新生成一个特征点并添加到特征点序列末尾
  传入的特征点指针feat是已经从特征点序列features中移除的，所以即使此特征点没有辅方向(第二个大于幅值阈值的方向)
  也会执行一次克隆feat，对其方向进行插值修正，并插入特征点序列的操作


  @param features new features are added to the end of this array
  @param hist orientation histogram
  @param n number of bins in hist
  @param mag_thr new features are added for entries in hist greater than this
  @param feat new features are clones of this with different orientations
*/
static void add_good_ori_features( CvSeq* features, double* hist, int n,
				   double mag_thr, struct feature* feat )
{
  struct feature* new_feat;
  double bin, PI2 = CV_PI * 2.0;
  int l, r, i;

  //遍历直方图
  for( i = 0; i < n; i++ )
    {
      l = ( i == 0 )? n - 1 : i-1;
      r = ( i + 1 ) % n;
      
	  //若当前的bin是局部极值(比前一个和后一个bin都大)，并且值大于给定的幅值阈值，则新生成一个特征点并添加到特征点序列末尾  
      if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
	{
	  bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
	  bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;//将插值结果规范到[0,n]内
	  new_feat = clone_feature( feat );//克隆当前特征点为新特征点
	  new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI;//新特征点的方向
	  cvSeqPush( features, new_feat );//插入到特征点序列末尾 
	  free( new_feat );
	}
    }
}



/*
  Makes a deep copy of a feature
  对输入的feature结构特征点做深拷贝，返回克隆生成的特征点的指针
  @param feat feature to be cloned

  @return Returns a deep copy of feat
*/
static struct feature* clone_feature( struct feature* feat )
{
  struct feature* new_feat;
  struct detection_data* ddata;

  new_feat = new_feature();
  ddata = feat_detection_data( new_feat );
  memcpy( new_feat, feat, sizeof( struct feature ) );
  memcpy( ddata, feat_detection_data(feat), sizeof( struct detection_data ) );
  new_feat->feature_data = ddata;

  return new_feat;
}



/*
  Computes feature descriptors for features in an array.  Based on Section 6
  of Lowe's paper.
  计算特征点序列中每个特征点的特征描述子向量

  @param features array of features
  @param gauss_pyr Gaussian scale space pyramid
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void compute_descriptors( CvSeq* features, IplImage*** gauss_pyr, int d,
				 int n )
{
  struct feature* feat;
  struct detection_data* ddata;
  double*** hist;//d*d*n的三维直方图数组
  int i, k = features->total;//特征点的个数 
  
  //遍历特征点序列中的特征点  
  for( i = 0; i < k; i++ )
    {
	  //调用宏，获取序列features中的第i个元素，并强制转换为struct feature类型  
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
	  //调用宏feat_detection_data来提取参数feat中的feature_data成员并转换为detection_data类型的指针
      ddata = feat_detection_data( feat );
	  //计算特征点附近区域的方向直方图，此直方图在计算特征描述子中要用到，返回值是一个d*d*n的三维数组 
      hist = descr_hist( gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
			 ddata->c, feat->ori, ddata->scl_octv, d, n );
      hist_to_descr( hist, d, n, feat );
      release_descr_hist( &hist, d );
    }
}



/*
  Computes the 2D array of orientation histograms that form the feature
  descriptor.  Based on Section 6.1 of Lowe's paper.

  @param img image used in descriptor computation
  @param r row coord of center of orientation histogram array
  @param c column coord of center of orientation histogram array
  @param ori canonical orientation of feature whose descr is being computed
  @param scl scale relative to img of feature whose descr is being computed
  @param d width of 2d array of orientation histograms
  @param n bins per orientation histogram

  @return Returns a d x d array of n-bin orientation histograms.
*/
static double*** descr_hist( IplImage* img, int r, int c, double ori,
			     double scl, int d, int n )
{
  double*** hist;
  double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
    grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
  int radius, i, j;

  //为直方图数组分配空间
  hist = calloc( d, sizeof( double** ) );//为第一维分配空间  
  for( i = 0; i < d; i++ )
    {
      hist[i] = calloc( d, sizeof( double* ) );//为第二维分配空间  
      for( j = 0; j < d; j++ )
	hist[i][j] = calloc( n, sizeof( double ) );//为第三维分配空间  
    }
  
  //为了保证特征描述子具有旋转不变性，要以特征点为中心，在附近邻域内旋转θ角，即旋转为特征点的方向  
  cos_t = cos( ori );
  sin_t = sin( ori );
  bins_per_rad = n / PI2;
  exp_denom = d * d * 0.5;
  hist_width = SIFT_DESCR_SCL_FCTR * scl;
  //考虑到要进行双线性插值，每个区域的宽度应为:SIFT_DESCR_SCL_FCTR * scl * ( d + 1.0 )  
  //在考虑到旋转因素，每个区域的宽度应为：SIFT_DESCR_SCL_FCTR * scl * ( d + 1.0 ) * sqrt(2)  
  //所以搜索的半径是：SIFT_DESCR_SCL_FCTR * scl * ( d + 1.0 ) * sqrt(2) / 2 
  radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;
  for( i = -radius; i <= radius; i++ )
    for( j = -radius; j <= radius; j++ )
      {
	/*
	  Calculate sample's histogram array coords rotated relative to ori.
	  Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
	  r_rot = 1.5) have full weight placed in row 1 after interpolation.
	*/
	c_rot = ( j * cos_t - i * sin_t ) / hist_width;
	r_rot = ( j * sin_t + i * cos_t ) / hist_width;
	rbin = r_rot + d / 2 - 0.5;
	cbin = c_rot + d / 2 - 0.5;
	
	if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
	  if( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori ))
	    {
	      grad_ori -= ori;
	      while( grad_ori < 0.0 )
		grad_ori += PI2;
	      while( grad_ori >= PI2 )
		grad_ori -= PI2;
	      
	      obin = grad_ori * bins_per_rad;
	      w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
	      interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n );
	    }
      }

  return hist;
}



/*
  Interpolates an entry into the array of orientation histograms that form
  the feature descriptor.

  @param hist 2D array of orientation histograms
  @param rbin sub-bin row coordinate of entry
  @param cbin sub-bin column coordinate of entry
  @param obin sub-bin orientation coordinate of entry
  @param mag size of entry
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void interp_hist_entry( double*** hist, double rbin, double cbin,
			       double obin, double mag, int d, int n )
{
  double d_r, d_c, d_o, v_r, v_c, v_o;
  double** row, * h;
  int r0, c0, o0, rb, cb, ob, r, c, o;

  r0 = cvFloor( rbin );
  c0 = cvFloor( cbin );
  o0 = cvFloor( obin );
  d_r = rbin - r0;
  d_c = cbin - c0;
  d_o = obin - o0;

  /*
    The entry is distributed into up to 8 bins.  Each entry into a bin
    is multiplied by a weight of 1 - d for each dimension, where d is the
    distance from the center value of the bin measured in bin units.
  */
  for( r = 0; r <= 1; r++ )
    {
      rb = r0 + r;
      if( rb >= 0  &&  rb < d )
	{
	  v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
	  row = hist[rb];
	  for( c = 0; c <= 1; c++ )
	    {
	      cb = c0 + c;
	      if( cb >= 0  &&  cb < d )
		{
		  v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
		  h = row[cb];
		  for( o = 0; o <= 1; o++ )
		    {
		      ob = ( o0 + o ) % n;
		      v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
		      h[ob] += v_o;
		    }
		}
	    }
	}
    }
}



/*
  Converts the 2D array of orientation histograms into a feature's descriptor
  vector.
  
  @param hist 2D array of orientation histograms
  @param d width of hist
  @param n bins per histogram
  @param feat feature into which to store descriptor
*/
static void hist_to_descr( double*** hist, int d, int n, struct feature* feat )
{
  int int_val, i, r, c, o, k = 0;

  for( r = 0; r < d; r++ )
    for( c = 0; c < d; c++ )
      for( o = 0; o < n; o++ )
	feat->descr[k++] = hist[r][c][o];

  feat->d = k;
  normalize_descr( feat );
  //遍历特征描述子向量，将超过阈值SIFT_DESCR_MAG_THR的元素强行赋值为SIFT_DESCR_MAG_THR
  for( i = 0; i < k; i++ )
    if( feat->descr[i] > SIFT_DESCR_MAG_THR )
      feat->descr[i] = SIFT_DESCR_MAG_THR;
  //再次归一化特征描述子向量 
  normalize_descr( feat );

  /* convert floating-point descriptor to integer valued descriptor */
  //遍历特征描述子向量，每个元素乘以系数SIFT_INT_DESCR_FCTR来变为整型，并且最大值不能超过255
  for( i = 0; i < k; i++ )
    {
      int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
      feat->descr[i] = MIN( 255, int_val );
    }
}



/*
  Normalizes a feature's descriptor vector to unitl length
  归一化特征点的特征描述子，即将特征描述子数组中每个元素除以特征描述子的模

  @param feat feature
*/
static void normalize_descr( struct feature* feat )
{
  double cur, len_inv, len_sq = 0.0;
  int i, d = feat->d;//特征描述子的维数 

  for( i = 0; i < d; i++ )
    {
      cur = feat->descr[i];
      len_sq += cur*cur;
    }
  len_inv = 1.0 / sqrt( len_sq );
  for( i = 0; i < d; i++ )
    feat->descr[i] *= len_inv;
}



/*
  Compares features for a decreasing-scale ordering.  Intended for use with
  CvSeqSort
  比较函数，将特征点按尺度的降序排列，用在序列排序函数CvSeqSort中

  @param feat1 first feature
  @param feat2 second feature
  @param param unused

  @return Returns 1 if feat1's scale is greater than feat2's, -1 if vice versa,
    and 0 if their scales are equal
*/
static int feature_cmp( void* feat1, void* feat2, void* param )
{
	//将输入的参数强制转换为struct feature类型的指针  
  struct feature* f1 = (struct feature*) feat1;
  struct feature* f2 = (struct feature*) feat2;

  //比较两个特征点的尺度值  
  if( f1->scl < f2->scl )
    return 1;
  if( f1->scl > f2->scl )
    return -1;
  return 0;
}



/*
  De-allocates memory held by a descriptor histogram
  释放计算特征描述子过程中用到的方向直方图的内存空间

  @param hist pointer to a 2D array of orientation histograms
  @param d width of hist
*/
static void release_descr_hist( double**** hist, int d )
{
  int i, j;

  for( i = 0; i < d; i++)
    {
      for( j = 0; j < d; j++ )
	free( (*hist)[i][j] );
      free( (*hist)[i] );
    }
  free( *hist );
  *hist = NULL;
}


/*
  De-allocates memory held by a scale space pyramid
  释放金字塔图像组的存储空间

  @param pyr scale space pyramid
  @param octvs number of octaves of scale space
  @param n number of images per octave
*/
static void release_pyr( IplImage**** pyr, int octvs, int n )
{
  int i, j;
  for( i = 0; i < octvs; i++ )
    {
      for( j = 0; j < n; j++ )
	cvReleaseImage( &(*pyr)[i][j] );
      free( (*pyr)[i] );
    }
  free( *pyr );
  *pyr = NULL;
}
