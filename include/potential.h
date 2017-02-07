#ifndef __potential
#define __potential

#include <stdio.h>
#include "feature.h"
#include "dataset.h"
#include "graph.h"

class LLearning;

//势函数父类
class LPotential
{
	protected :
		int classNo;
		const char *trainFolder, *trainFile, *evalFolder, *evalExtension;
	public :
		LDataset *dataset;
		LCrf *crf;
		LCrfDomain *domain;
		int nodeOffset, edgeOffset;

		LPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo);
		virtual ~LPotential() {};

		void Evaluate(LList<char *> &imageFiles, int from = 0, int to = -1);

		virtual int GetNodeCount();
		virtual int GetEdgeCount();

		virtual void Train(LList<char *> &trainImageFiles) {};
		virtual void SaveTraining() {};
		virtual void LoadTraining() {};
		virtual void Evaluate(char *imageFileName) {};
		virtual void Initialize(LLabImage &labImage, char *imageFileName) {};
		virtual void UnInitialize() {};
		virtual void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes) {};
		virtual double GetCost(LCrfDomain *costDomain) { return(0); }
};

//一阶一元势函数
class LUnaryPixelPotential : public LPotential
{
	protected :
		double unaryFactor;
		int width, height, unaryCount;
		//数据项
		double *unaryCosts;
		LBaseCrfLayer *layer;
	public :
		LUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor);
		~LUnaryPixelPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void SetLabels();
		virtual double GetCost(LCrfDomain *costDomain);
};

//视差模型一阶一元势函数
class LDisparityUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		double filterSigma, maxDistance, distanceBeta;
		int subSample, maxDelta;
		const char *leftFolder, *rightFolder;
	public :
		LDisparityUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setUnaryFactor, double setFilterSigma, double setMaxDistance, double setDistanceBeta, int setSubSample, int setMaxDelta, const char *setLeftFolder, const char *setRightFolder);
		~LDisparityUnaryPixelPotential();
		//没有训练
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//基于DenseFeature的一阶一元势函数
class LDenseUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		//形状
		struct LShape
		{
			int x, y, width, height;
		};
		int thetaIncrement, thetaStart, numberOfThetas;

	private :
		//特征的整数图的指针，特征数*每个维数，所以是一个三维指针
		LIntegralImage ***integralImages;
		//最小最大矩形框
		int minimumRectangleSize, maximumRectangleSize;
		//降采样率等
		int subSample, *buckets, fileSum, **featureValues, numberOfShapes, *integralBuckets, *integralPoints;
		//权重
		double **weights, *classWeights;
		int *targets, totalFeatures;
		//有效点
		int **validPointsX, **validPointsY, *pointsSum, numExamples;
		LShape *shapes;
		double maxClassRatio;
		//特征
		LList<LFeature *> features;

		//计算形状滤波
		int CalculateShapeFilterResponse(int index, int bucket, LShape *shape, int pointX, int pointY);
		//初始化训练数据
		void InitTrainData(LList<char *> &trainImageFiles);
		void InitTrainData(LLabImage *labImages, LLabelImage *groundTruth);
		//反初始化
		void UnInitTrainData();
		//初始化测试数据
		void InitEvalData(char *evalImageFile, int *width, int *height);
		void InitEvalData(LLabImage &labImage);
		//反初始化
		void UnInitEvalData();
		//计算整数图
		void CalculateIntegralImages(LLabImage &labImage, LIntegralImage ***integralImage, int subSample, int *width, int *height, char *imageFileName);
	public :
		//学习器
		LLearning *learning;

		LDenseUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor, int setSubSample, int setNumberOfShapes, int setMinimumRectangleSize, int setMaximumRectangleSize, double setMaxClassRatio);
		~LDenseUnaryPixelPotential();

		void AddFeature(LDenseFeature *feature);

		//训练
		void Train(LList<char *> &trainImageFiles);
		void Train(LLabImage *labImages, LLabelImage *groundTruth, int count);

		void SaveTraining();
		void LoadTraining();
		//测试
		void Evaluate(char *imageFileName);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		int *GetTrainBoostingValues(int index, int core);
		int *GetEvalBoostingValues(int index);
		int *GetTrainForestValues(int index, int *pixelIndexes, int count, int core);
		int GetEvalForestValue(int index, int pixelIndex);
		int GetLength(int index);
		int GetSize(int index);
};

//基于目标高度的势函数，连接目标检测和3D重建的数据项
class LHeightUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		//相机参数
		double cameraBaseline, cameraHeight, cameraFocalLength, cameraAspectRatio, cameraWidthOffset, cameraHeightOffset;
		//最小最大值和阈值
		double minHeight, maxHeight, threshold;
		//直方图需要聚类，即根据高度分布直方图，聚类到某个目标
		int numberOfClusters, disparityClassNo;
		int *heightCounts, *totals, subSample;
		//包含的模型
		LCrfDomain *objDomain, *dispDomain;
		LBaseCrfLayer *dispLayer, *objLayer;
		int first;
	public :
		LHeightUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setDispDomain, LBaseCrfLayer *setDispLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setUnaryFactor, int setDisparityClassNo, double setCameraBaseline, double setCameraHeight, double setCameraFocalLength, double setCameraAspectRatio, double setCameraWidthOffset, double setCameraHeightOffset, double setMinHeight, double setMaxHeight, int setNumberOfClusters, double setThreshold, int setSubSample);
		~LHeightUnaryPixelPotential();
		//添加代价
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);

		void Initialize(LLabImage &labImage, char *imageFileName);
		//这个是要训练的，根据训练集统计目标高度分布
		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();
		void UnInitialize();
		double GetCost(LCrfDomain *costDomain);
};

//一阶二元势函数
class LPairwisePixelPotential : public LPotential
{
	protected :
		LBaseCrfLayer *layer;
	public :
		LPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo);
};

//基于potts模型的一阶二元势函数，即相同无代价，不同有代价
class LPottsPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
	public :
		LPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo);
		~LPottsPairwisePixelPotential();
		int GetEdgeCount();
		//把代价值添加到图模型中
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//八邻域的一阶二元势函数
class LEightNeighbourPottsPairwisePixelPotential : public LPottsPairwisePixelPotential
{
	protected :
		double pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight;
		//邻域差异
		double PairwiseDiff(double *lab1, double *lab2, double distance);
	public :
		LEightNeighbourPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight);
		//不用训练
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//基于不一致性线性截断的一阶二元势函数
class LLinearTruncatedPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
		double pairwiseFactor;
		double truncation;
	public :
		LLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation);
		~LLinearTruncatedPairwisePixelPotential();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//基于视差不一致性线性截断的一阶二元势函数
class LDisparityLinearTruncatedPairwisePixelPotential : public LLinearTruncatedPairwisePixelPotential
{
	protected :
	public :
		LDisparityLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation);
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//联合目标检测和3D重建的二元势函数
class LJointPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
		int first;

		double *objCosts, disparityFactor, crossFactor, truncation;
		double pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight;
		int disparityClassNo;
		double PairwiseDiff(double *lab1, double *lab2, double distance);

		LCrfDomain *objDomain, *dispDomain;
		LBaseCrfLayer *dispLayer, *objLayer;
	public :
		LJointPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setDispDomain, LBaseCrfLayer *setDispLayer, int setClassNo, int setDisparityClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight, double setDisparityFactor, double setTruncation, double setCrossFactor);
		~LJointPairwisePixelPotential();

		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//基于segment的一元势函数
class LUnarySegmentPotential : public LPotential
{
	protected :
		double consistencyPrior, segmentFactor;
		LList<LPnCrfLayer *> layers;
	public :
		double *unaryCosts;

		LUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor);
		~LUnarySegmentPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void AddLayer(LPnCrfLayer *layer);
		double GetCost(LCrfDomain *costDomain);
};

//鼓励segment内一致性的一元势函数
class LConsistencyUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
	public :
		LConsistencyUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, int setClassNo, double setConsistencyPrior);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
};

//高阶层的数函数类
class LStatsUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
		double *clusterProbabilities, minLabelRatio, kMeansMaxChange, alpha, maxClassRatio;
		int colourStatsClusters, pointsPerKdCluster, *buckets, *buckets2;

		int numExamples, totalFeatures, *targets, totalSegments, width, height;
		double **weights, **featureValues;
		int numberOfThetas;
		double *classWeights, **evalData;
		LSegmentation2D *segmentation;
		LList<LDenseFeature *> features;
		int neighbour;
	public :
		LLearning *learning;

		LStatsUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor, double setMinLabelRatio, double setAlpha, double setMaxClassRatio, int setNeighbour = 0, LSegmentation2D *setSegmentation = NULL);
		~LStatsUnarySegmentPotential();

		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();
		void Evaluate(char *imageFileName);
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		void AddFeature(LDenseFeature *feature);

		double *GetTrainBoostingValues(int index, int core);
		double *GetEvalBoostingValues(int index);
		double *GetTrainSVMValues(int index, int core);
		double *GetEvalSVMValues(int index);
};

//基于segment的二元势函数
class LPairwiseSegmentPotential : public LPotential
{
	protected :
		LPnCrfLayer *layer;
	public :
		LPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo);
		//不用训练
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
};

//基于Potts模型的高阶二元势函数
class LPottsPairwiseSegmentPotential : public LPairwiseSegmentPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
	public :
		LPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo);
		~LPottsPairwiseSegmentPotential();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//基于特征直方图的高阶二元势函数
class LHistogramPottsPairwiseSegmentPotential : public LPottsPairwiseSegmentPotential
{
	protected :
		double pairwisePrior, pairwiseFactor, pairwiseBeta;
		int buckets;
		//计算直方图欧氏距离
		double PairwiseDistance(double *h1, double *h2);
	public :
		LHistogramPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, int setBuckets);
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//基于图像的一元势函数
class LUnaryImagePotential : public LPotential
{
	protected :
		double *unaryImageFactor;
		LPreferenceCrfLayer *layer;
		double *unaryCosts;
	public :
		LUnaryImagePotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPreferenceCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double *setUnaryImageFactor);
		~LUnaryImagePotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int GetEdgeCount();
		double GetCost(LCrfDomain *costDomain);
};

//基于图像的二元势函数
class LPairwiseImagePotential : public LPotential
{
	protected :
		double pairwiseImageFactor, *pairwiseCosts;
		LPreferenceCrfLayer *layer;
	public :
		LPairwiseImagePotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPreferenceCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setpairwiseImageFactor);
		~LPairwiseImagePotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int GetEdgeCount();
		double GetCost(LCrfDomain *costDomain);
};

//基于目标共生信息的势函数
class LCooccurencePairwiseImagePotential : public LPairwiseImagePotential
{
	protected :
		//目标总数，共同出现的次数，训练样本总数
		int *classOccurence, *classCooccurence, total;
	public :
		LCooccurencePairwiseImagePotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPreferenceCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setPairwiseImageFactor);
		~LCooccurencePairwiseImagePotential();

		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//Dummy专用,应该是结合目标检测与图像分割的方法
//一元势函数
class LDummyUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
	public :
		LDummyUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//二元势函数
class LDummyUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
		double alpha;
	public :
		LDummyUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setSegmentFactor, double setAlpha);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//检测器势函数
class LDummyDetectorPotential : public LUnarySegmentPotential
{
	protected :
		double detectorThreshold;
	public :
		LDummyDetectorPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setSegmentFactor, double setDetectorThreshold);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

#endif