#ifndef __dataset
#define __dataset

#include <stdio.h>
#include "std.h"

class LCrf;
class LCrfDomain;
class LCrfLayer;
class LBaseCrfLayer;
class LPnCrfLayer;
class LPreferenceCrfLayer;
class LLabelImage;

//数据集
class LDataset
{
	private :
		//文件夹名
		char *GetFolderFileName(const char *imageFile, const char *folder, const char *extension);
		//字符串排序
		static int SortStr(char *str1, char *str2);
	protected :
		//载入文件夹,把数据都载入List中
		void LoadFolder(const char *folder, const char *extension, LList<char *> &list);
	public :
		LDataset();
		virtual ~LDataset();
		virtual void Init();
		//存放训练文件，测试文件，所有文件名的
		LList<char *> trainImageFiles, testImageFiles, allImageFiles;

		unsigned int seed;
		int classNo, filePermutations, featuresOnline;
		double proportionTrain, proportionTest;

		const char *imageFolder, *imageExtension, *groundTruthFolder, *groundTruthExtension, *trainFolder, *testFolder, *dispTestFolder;
		int optimizeAverage;

		//K-means参数,用于特征聚类
		int clusterPointsPerKDTreeCluster;
		double clusterKMeansMaxChange;
		//location
		int locationBuckets;
		const char *locationFolder, *locationExtension;
		//TextonBoost
		int textonNumberOfClusters, textonKMeansSubSample;
		double textonFilterBankRescale;
		const char *textonClusteringTrainFile, *textonFolder, *textonExtension;
		//Sift
		int siftKMeansSubSample, siftNumberOfClusters, siftWindowSize, siftWindowNumber, sift360, siftAngles, siftSizeCount, siftSizes[4];
		const char *siftClusteringTrainFile, *siftFolder, *siftExtension;
		//颜色Sift
		int colourSiftKMeansSubSample, colourSiftNumberOfClusters, colourSiftWindowSize, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftSizeCount, colourSiftSizes[4];
		const char *colourSiftClusteringTrainFile, *colourSiftFolder, *colourSiftExtension;
		//LBP
		int lbpSize, lbpKMeansSubSample, lbpNumberOfClusters;
		const char *lbpClusteringFile, *lbpFolder, *lbpExtension;
		//Boosting分类器的参数
		int denseNumRoundsBoosting, denseBoostingSubSample, denseNumberOfThetas, denseThetaStart, denseThetaIncrement, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize;
		double denseRandomizationFactor, denseWeight, denseMaxClassRatio;
		const char *denseBoostTrainFile, *denseExtension, *denseFolder;
		//Mean-Sift参数，用于像素聚类
		double meanShiftXY[4], meanShiftLuv[4];
		int meanShiftMinRegion[4];
		const char *meanShiftFolder[4], *meanShiftExtension;
		//一致性先验项
		double consistencyPrior;
		//segment Boosting参数
		int statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsNumberOfBoosts;
		double statsRandomizationFactor, statsAlpha, statsFactor, statsPrior, statsMaxClassRatio;
		const char *statsTrainFile, *statsExtension, *statsFolder;
		//权重设置
		double pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, pairwisePrior, pairwiseFactor, pairwiseBeta;
		double cliqueMinLabelRatio, cliqueThresholdRatio, cliqueTruncation;

		int pairwiseSegmentBuckets;
		double pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta;
		//共生信息
		const char *cooccurenceTrainFile;
		double cooccurenceWeight;
		//双目视觉
		double disparityUnaryFactor, disparityFilterSigma, disparityMaxDistance, disparityDistanceBeta;
		int disparitySubSample, disparityMaxDelta, disparityClassNo, disparityRangeMoveSize;
		double disparityPairwiseFactor, disparityPairwiseTruncation;
		const char *disparityLeftFolder, *disparityRightFolder, *disparityGroundTruthFolder, *disparityGroundTruthExtension;

		double cameraBaseline, cameraHeight, cameraFocalLength, cameraAspectRatio, cameraWidthOffset, cameraHeightOffset;
		double crossUnaryWeight, crossPairwiseWeight, crossMinHeight, crossMaxHeight, crossThreshold;
		int crossHeightClusters;
		const char *crossTrainFile;
		//K-means参数
		double kMeansXyLuvRatio[6];
		const char *kMeansFolder[6], *kMeansExtension;
		int kMeansIterations, kMeansMaxDiff, kMeansDistance[6];

		int unaryWeighted;
		double *unaryWeights;
		//标签图像转换
		virtual void RgbToLabel(unsigned char *rgb, unsigned char *label);
		virtual void LabelToRgb(unsigned char *label, unsigned char *rgb);
		//保存图像
		virtual void SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName);
		//创建CRF模型
		virtual void SetCRFStructure(LCrf *crf) {};
		//无监督图像分割
		virtual int Segmented(char *imageFileName);
		//获取标签配置
		virtual void GetLabelSet(unsigned char *labelset, char *imageFileName);
};

class LSowerbyDataset : public LDataset
{
	// sky      0, grass   1, roadline 2, road    3
	// building 4, sign    5, car      6

	private :
	public :
		LSowerbyDataset();
		void SetCRFStructure(LCrf *crf);
};

//这个数据集分辨率较低
class LCorelDataset : public LDataset
{
	// rhino/hippo 0, polarbear 1, water 2, snow    3
	// vegetation  4, ground    5, sky   6

	private :
	public :
		LCorelDataset();
		void SetCRFStructure(LCrf *crf);
};

//微软的目标检测和分割数据集
class LMsrcDataset : public LDataset
{
	// building  0, grass     1, tree     2, cow       3
	// horse     4, sheep     5, sky      6, mountain  7
	// plane     8, water     9, face    10, car      11
	// bike     12, flower   13, sign    14, bird     15
	// book     16, chair    17, road    18, cat      19
	// dog      20, body     21, boat    22

	private :
	protected :
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
	public :
		const char *trainFileList, *testFileList;
		void Init();

		LMsrcDataset();
		void SetCRFStructure(LCrf *crf);
};

//双目数据集
class LLeuvenDataset : public LDataset
{
	// building 0, tree     1, sky         2, car   3
	// sign     4, road     5, pedestrian  6, fense 7
	// column   8, pavement 9, bicyclist  10,
	private :
		void AddFolder(char *folder, LList<char *> &fileList);
	protected :
		//目标类别标签和图像的转换
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
	public :
		void Init();

		LLeuvenDataset();
		//创建CRF模型
		void SetCRFStructure(LCrf *crf);

		//视差值和图像的转换
		void DisparityRgbToLabel(unsigned char *rgb, unsigned char *label);
		void DisparityLabelToRgb(unsigned char *label, unsigned char *rgb);
};

//Pascal VOC目标检测和分割数据集
class LVOCDataset : public LDataset
{
	// background  0, aeroplane  1, bicycle    2, bird       3
	// boat        4, bottle     5, bus        6, car        7
	// cat         8, chair      9, cow       10, din.table 11
	// dog        12, horse     13, motorbike 14, person    15
	// plant      16, sheep     17, sofa      18, train     19
	// tv-monitor 20

	private :
	protected :
	public :
		const char *trainFileList, *testFileList;
		void Init();

		LVOCDataset();
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
		void SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName);
		void SetCRFStructure(LCrf *crf);
};

//单目道路数据集
class LCamVidDataset : public LDataset
{
	// building 0, tree     1, sky         2, car   3
	// sign     4, road     5, pedestrian  6, fense 7
	// column   8, pavement 9, bicyclist  10,

	private :
		void AddFolder(char *folder, LList<char *> &fileList);
	protected :
		void Init();
	public :
		LCamVidDataset();
		
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
		void SetCRFStructure(LCrf *crf);
};

/////////////////////////////////////////////////////////////////////////////////////////////
//
//				KITTI	Dataet
//
/////////////////////////////////////////////////////////////////////////////////////////////

//KITTI道路检测数据集
class LKITTIDataset : public LDataset
{
	//0 non-road	1 road

private:
public:

	LKITTIDataset();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};


#endif