#include <getpsnr.h>
using namespace cv;
double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|AbsDiff������ OpenCV �м������������ľ���ֵ�ĺ���
	s1.convertTo(s1, CV_32F);  // ��������ʹ�õ�CV_32F�����㣬��Ϊ8λ�޷���char�ǲ��ܽ���ƽ������
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         //��ÿһ��ͨ�����мӺ�

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // ���ڷǳ�С��ֵ���ǽ�Լ����0
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());//����MSE
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;//����PSNR
	}
}