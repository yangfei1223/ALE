#ifndef __image
#define __image

#include "filter.h"
#include "std.h"

class LCrfDomain;
class LDataset;
class LGreyImage;
class LRgbImage;
class LLuvImage;
class LLabImage;
class LLabelImage;
class LCostImage;
class LSegmentImage;
template <class T> class LFilter2D;

template <class T>

//图像类，T为每个像素的类型，可根据不同需求设置不同位数的图像
class LImage
{
	protected :
		T *data;	//image data
		int width, height, bands;	//bands应该是通道数
	public :
		//一些构造函数
		LImage();
		LImage(int setBands);
		LImage(int setWidth, int setHeight, int setBands);
		LImage(LImage<T> &image);
		LImage(char *fileName);
		~LImage();

		//设置分辨率
		void SetResolution(int setWidth, int setHeight);
		void SetResolution(int setWidth, int setHeight, int setBands);
		//拷贝数据
		//拷贝整个图像数据
		void CopyDataFrom(LImage<T> &image);
		//只拷贝数据
		void CopyDataFrom(T *fromData);
		//拷贝数据到
		void CopyDataTo(T *toData);

		//返回图像的指针
		T *GetData();
		//迭代器，返回的是指向像素的指针
		T *operator()(int x, int y);
		//精确到通道
		T &operator()(int x, int y, int band);
		//返回像素值
		T &GetValue(int index);
		
		//获取一些图像属性
		int GetWidth();
		int GetHeight();
		int GetPoints();
		int GetBands();
		int GetSize();

		//窗口拷贝
		void WindowTo(LImage &imageTo, int centreX, int centreY, int halfWidth, int halfHeight);
		void WindowFrom(LImage &imageTo, int centreX, int centreY, int halfWidth, int halfHeight);

		//读图像
		virtual void Save(char *fileName);
		//写图像
		virtual void Load(char *fileName);
		//判断文件是否存在
		static int Exist(char *fileName);
};

template <class T>
class LColourImage : public LImage<T>
{
	public :
		LColourImage();
		LColourImage(int setBands);
		LColourImage(int setWidth, int setHeight, int setBands);
		LColourImage(LColourImage<T> &image);

		void FilterFrom(LColourImage<T> &fromImage, LFilter2D<T> &filter);
		void FilterTo(LColourImage<T> &toImage, LFilter2D<T> &filter);
		void ScaleTo(LColourImage &imageTo, double scale);
		void ScaleFrom(LColourImage &imageFrom, double scale);
		void ScaleTo(LColourImage &imageTo, double centreX, double centreY, double scale, int halfWidth, int halfHeight);
		void ScaleFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, int halfWidth, int halfHeight);
		void RotateTo(LColourImage &imageTo, double centreX, double centreY, double angle, int halfWidth, int halfHeight);
		void RotateFrom(LColourImage &imageFrom, double centreX, double centreY, double angle, int halfWidth, int halfHeight);
		void ScaleRotateTo(LColourImage &imageTo, double centreX, double centreY, double scale, double angle, int halfWidth, int halfHeight);
		void ScaleRotateFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, double angle, int halfWidth, int halfHeight);
		void AffineTo(LColourImage &imageTo, double centreX, double centreY, double scale, double *u, int halfWidth, int halfHeight);
		void AffineFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, double *u, int halfWidth, int halfHeight);
		void AffineRotateTo(LColourImage &imageTo, double centreX, double centreY, double scale, double *u, double angle, int halfWidth, int halfHeight);
		void AffineRotateFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, double *u, double angle, int halfWidth, int halfHeight);
};

class LRgbImage : public LColourImage<unsigned char>
{
	public :
		//颜色空间转换
		static void RgbToGrey(unsigned char *rgb, double *grey);
		static void RgbToLuv(unsigned char *rgb, double *luv);
		static void RgbToLab(unsigned char *rgb, double *lab);

		//一些构造函数
		LRgbImage();
		LRgbImage(int setWidth, int setHeight);
		LRgbImage(char *fileName);
		LRgbImage(LRgbImage &rgbImage);
		LRgbImage(LGreyImage &greyImage);
		LRgbImage(LLuvImage &luvImage);
		LRgbImage(LLabImage &labImage);
		LRgbImage(LLabelImage &labelImage, LCrfDomain *domain);
		LRgbImage(LLabelImage &labelImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *));

		LRgbImage(LSegmentImage &segmentImage, LRgbImage &rgbImage, int showBoundaries);
		LRgbImage(LCostImage &costImage, LCrfDomain *domain, int showMaximum);

		void Load(char *fileName);
		void Load(LRgbImage &rgbImage);
		void Load(LGreyImage &greyImage);
		void Load(LLuvImage &luvImage);
		void Load(LLabImage &labImage);
		void Load(LLabelImage &labelImage, LCrfDomain *domain);
		void Load(LLabelImage &labelImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *));
		void Load(LSegmentImage &segmentImage, LRgbImage &rgbImage, int showBoundaries);
		void Load(LCostImage &costImage, LCrfDomain *domain, int showMaximum);

		void Save(char *fileName);
		void Save(LRgbImage &rgbImage);
		void Save(LGreyImage &greyImage);
		void Save(LLuvImage &luvImage);
		void Save(LLabImage &labImage);
		void Save(LLabelImage &labelImage, LCrfDomain *domain);
		void Save(LLabelImage &labelImage, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *));
};

class LGreyImage : public LColourImage<double>
{
	public :
		static void GreyToRgb(double *grey, unsigned char *rgb);
		static void GreyToLuv(double *grey, double *luv);
		static void GreyToLab(double *grey, double *lab);

		LGreyImage();
		LGreyImage(int setWidth, int setHeight);
		LGreyImage(char *fileName);
		LGreyImage(LRgbImage &rgbImage);
		LGreyImage(LGreyImage &greyImage);
		LGreyImage(LLuvImage &luvImage);
		LGreyImage(LLabImage &labImage);

		void Load(char *fileName);
		void Load(LRgbImage &rgbImage);
		void Load(LGreyImage &greyImage);
		void Load(LLuvImage &luvImage);
		void Load(LLabImage &labImage);

		void Save(char *fileName);
		void Save(LRgbImage &rgbImage);
		void Save(LGreyImage &greyImage);
		void Save(LLuvImage &luvImage);
		void Save(LLabImage &labImage);
};

class LLuvImage : public LColourImage<double>
{
	public :
		static void LuvToRgb(double *luv, unsigned char *rgb);
		static void LuvToGrey(double *luv, double *grey);

		LLuvImage();
		LLuvImage(int setWidth, int setHeight);
		LLuvImage(char *fileName);
		LLuvImage(LRgbImage &rgbImage);
		LLuvImage(LGreyImage &greyImage);
		LLuvImage(LLuvImage &luvImage);
		LLuvImage(LLabImage &labImage);

		void Load(char *fileName);
		void Load(LRgbImage &rgbImage);
		void Load(LGreyImage &greyImage);
		void Load(LLuvImage &luvImage);
		void Load(LLabImage &labImage);

		void Save(char *fileName);
		void Save(LRgbImage &rgbImage);
		void Save(LGreyImage &greyImage);
		void Save(LLuvImage &luvImage);
		void Save(LLabImage &labImage);
};

class LLabImage : public LColourImage<double>
{
	public :
		static void LabToGrey(double *luv, double *grey);
		static void LabToRgb(double *lab, unsigned char *rgb);

		LLabImage();
		LLabImage(int setWidth, int setHeight);
		LLabImage(char *fileName);
		LLabImage(LRgbImage &rgbImage);
		LLabImage(LGreyImage &greyImage);
		LLabImage(LLabImage &labImage);
		LLabImage(LLuvImage &luvImage);

		void Load(char *fileName);
		void Load(LRgbImage &rgbImage);
		void Load(LGreyImage &greyImage);
		void Load(LLuvImage &luvImage);
		void Load(LLabImage &labImage);

		void Save(char *fileName);
		void Save(LRgbImage &rgbImage);
		void Save(LGreyImage &greyImage);
		void Save(LLuvImage &luvImage);
		void Save(LLabImage &labImage);
};

class LLabelImage : public LImage<unsigned char>
{
	public :
		LLabelImage();
		LLabelImage(int setWidth, int setHeight);
		LLabelImage(char *fileName, LCrfDomain *domain);
		LLabelImage(char *fileName, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *));
		LLabelImage(LRgbImage &rgbImage, LCrfDomain *domain);
		LLabelImage(LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *));
		LLabelImage(LLabelImage &labelImage);
		LLabelImage(LCostImage &costImage, int showMaximum);

		void Load(char *fileName, LCrfDomain *domain);
		void Load(char *fileName, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *));
		void Load(LRgbImage &rgbImage, LCrfDomain *domain);
		void Load(LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *));
		void Load(LLabelImage &labelImage);
		void Load(LCostImage &costImage, int showMaximum);

		void Save(char *fileName, LCrfDomain *domain);
		void Save(char *fileName, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *));
		void Save(char *fileName, LRgbImage &rgbImage, LCrfDomain *domain);
		void Save(char *fileName, LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *));
		void Save(LRgbImage &rgbImage, LCrfDomain *domain);
		void Save(LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *));

		void Save(LLabelImage &labelImage);
		void Save8bit(char *fileName);
};

class LCostImage : public LImage<double>
{
	private :
		void CostToLabel(double *cost, unsigned char *label, int showMaximum);
	public :
		LCostImage();
		LCostImage(LCostImage &costImage);
		LCostImage(int setWidth, int setHeight, int setBands);

		void Save(char *fileName, LCrfDomain *domain, int showMaximum);
		void Save(LRgbImage &rgbImage, LCrfDomain *domain, int showMaximum);
		void Save(LLabelImage &labelImage, int showMaximum);
};

//分割图像
class LSegmentImage : public LImage<int>
{
	public :
		LSegmentImage();
		LSegmentImage(LSegmentImage &segmentImage);
		LSegmentImage(int setWidth, int setHeight);

		//保存
		void Save(char *fileName, LRgbImage &rgbImage, int showBoundaries);
		void Save(LRgbImage &segmentRgbImage, LRgbImage &rgbImage, int showBoundaries);
};

class LIntegralImage
{
	protected :
	public :
		int width, height;
		virtual ~LIntegralImage() {};
	public :
		LIntegralImage();
		virtual int Response(int x1, int y1, int x2, int y2) { return(0); };
		virtual double DResponse(double x1, double y1, double x2, double y2) { return(0); };
		virtual void Load(LImage<unsigned short> &dataImage, int subSample, int index) {};
		virtual void Copy(LImage<double> &dataImage, int subSample, int index, double scale) {};
};

class LIntegralImage4B : public LIntegralImage
{
	private :
		LImage <unsigned int> image;
	public :
		int Response(int x1, int y1, int x2, int y2);
		double DResponse(double x1, double y1, double x2, double y2);
		void Load(LImage<unsigned short> &dataImage, int subSample, int index);
		void Copy(LImage<double> &dataImage, int subSample, int index, double scale);
};

class LIntegralImage2B : public LIntegralImage
{
	private :
		LImage <unsigned short> image;
	public :
		int Response(int x1, int y1, int x2, int y2);
		double DResponse(double x1, double y1, double x2, double y2);
		void Load(LImage<unsigned short> &dataImage, int subSample, int index);
		void Copy(LImage<double> &dataImage, int subSample, int index, double scale);
};

class LIntegralImage1B : public LIntegralImage
{
	private :
		LImage <unsigned char> image;
	public :
		int Response(int x1, int y1, int x2, int y2);
		double DResponse(double x1, double y1, double x2, double y2);
		void Load(LImage<unsigned short> &dataImage, int subSample, int index);
		void Copy(LImage<double> &dataImage, int subSample, int index, double scale);
};

class LIntegralImageHB : public LIntegralImage
{
	private :
		LImage <unsigned char> image;
	public :
		int Response(int x1, int x2, int y1, int y2);
		double DResponse(double x1, double y1, double x2, double y2);
		void Load(LImage<unsigned short> &dataImage, int subSample, int index);
};

#endif