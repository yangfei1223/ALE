#ifndef __feature
#define __feature

#include "std.h"
#include "image.h"
#include "clustering.h"
#include "segmentation.h"

//特征父类
class LFeature
{
	protected :
		//数据集
		LDataset *dataset;
		//文件夹名和文件名
		const char *trainFolder, *trainFile;
	public :
		//评估文件名和评价扩展名
		const char *evalFolder, *evalExtension;

		//构造函数
		LFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension);
		virtual ~LFeature() {};
		//评估
		void Evaluate(LList<char *> &imageFiles, int from = 0, int to = -1);

		//载入
		virtual void LoadTraining() = 0;
		//保存
		virtual void SaveTraining() = 0;
		//训练
		virtual void Train(LList<char *> &trainImageFiles) = 0;
		virtual int GetBuckets() = 0;
		virtual void Evaluate(char *imageFileName) {};
};

//像素特征类
class LDenseFeature : public LFeature
{
	public :
		LDenseFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension);

		virtual void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName) = 0;
		virtual void Evaluate(char *imageFileName);
};

//纹理特征
class LTextonFeature : public LDenseFeature
{
	private :
		//不同的滤波器
		LList<LFilter2D<double> *> filters;
		//滤波器以及聚类操作的参数
		int subSample, numberOfClusters, pointsPerKDTreeCluster;
		double filterScale, kMeansMaxChange;
		//聚类器
		LKMeansClustering<double> *clustering;

		//创建滤波器
		void CreateFilterList();
	public :
		//构造函数
		LTextonFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, double setFilterScale, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster);

		~LTextonFeature();
		//获取滤波操作总数，注意每个通道都要滤波
		int GetTotalFilterCount();

		//训练
		void Train(LList<char *> &trainImageFiles);
		//离散化
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		//载入训练数据
		void LoadTraining();
		//保存训练数据
		void SaveTraining();
		int GetBuckets();
};

class LSiftFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, *windowSize, windowNumber, is360, angles, diff, sizeCount;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		~LSiftFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LColourSiftFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, *windowSize, windowNumber, is360, angles, diff, sizeCount;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		~LColourSiftFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LLbpFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, windowSize;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LLbpFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster);
		~LLbpFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

//位置特征类
class LLocationFeature : public LDenseFeature
{
	private :
		//栅格聚类，位置特征使用的是栅格聚类方法
		LLatticeClustering<double> *clustering;
	public :
		//构造函数
		LLocationFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setLocationBuckets);
		~LLocationFeature();

		//特征提取，位置特征只需要在测试时提取
		void Train(LList<char *> &trainImageFiles) { printf("Location need not to train.\n"); };
		//离散化
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining() {};
		void SaveTraining() {};
		int GetBuckets();
};

class LDummyFeature : public LDenseFeature
{
	private :
		int numberOfClusters;
	public :
		LDummyFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setNumberOfClusters);

		void Evaluate(char *imageFileName) {};
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName) {};
		void Train(LList<char *> &trainImageFiles) { printf("Dummy nend not to train.\n"); };
		void LoadTraining() {};
		void SaveTraining() {};
		int GetBuckets();
};

#endif