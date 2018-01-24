#include <getpsnr.h>
using namespace cv;
double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|AbsDiff函数是 OpenCV 中计算两个数组差的绝对值的函数
	s1.convertTo(s1, CV_32F);  // 这里我们使用的CV_32F来计算，因为8位无符号char是不能进行平方计算
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         //对每一个通道进行加和

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // 对于非常小的值我们将约等于0
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());//计算MSE
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;//返回PSNR
	}
}