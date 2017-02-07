#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#include <fnmatch.h>
#endif

#include "../include/dataset.h"
#include "../include/potential.h"
#include "../include/crf.h"

LDataset::LDataset()
{
}

LDataset::~LDataset()
{
	int i;
	for (i = 0; i < allImageFiles.GetCount(); i++) delete[] allImageFiles[i];
}

void LDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);
}

void LDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	rgb[0] = rgb[1] = rgb[2] = 0;
	for (int i = 0; lab > 0; i++, lab >>= 3)
	{
		rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
		rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
		rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
	}
}

char *LDataset::GetFolderFileName(const char *imageFile, const char *folder, const char *extension)
{
	char *fileName;
	fileName = new char[strlen(imageFile) + strlen(folder) - strlen(imageFolder) + strlen(extension) - strlen(imageExtension) + 1];
	strcpy(fileName, folder);
	strncpy(fileName + strlen(folder), imageFile + strlen(imageFolder), strlen(imageFile) - strlen(imageFolder) - strlen(imageExtension));
	strcpy(fileName + strlen(imageFile) + strlen(folder) - strlen(imageFolder) - strlen(imageExtension), extension);
	return(fileName);
}

int LDataset::SortStr(char *str1, char *str2)
{
	return(strcmp(str1, str2));
};

void LDataset::LoadFolder(const char *folder, const char *extension, LList<char *> &list)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(folder) + strlen(extension) + 2];
	sprintf(folderExt, "%s*%s", folder, extension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(extension)] = 0;
		fileName = new char[strlen(info.name) + 1];
		strcpy(fileName, info.name);
		list.Add(fileName);
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(extension) + 2];
	sprintf(folderExt, "*%s", extension);

	count = scandir(folder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(extension)] = 0;
				fileName = new char[strlen(nameList[i]->d_name) + 1];
				strcpy(fileName, nameList[i]->d_name);
				list.Add(fileName);
			}
			if (nameList[i] != NULL) free(nameList[i]);
		}
		if (nameList != NULL) free(nameList);
	}
#endif
	delete[] folderExt;
}

void LDataset::Init()
{
	int index1, index2, i;

	//随机种子
	LMath::SetSeed(seed);

	LoadFolder(imageFolder, imageExtension, allImageFiles);

	//打乱顺序，filePermutations是随机交换次数
	printf("Permuting image files..\n");
	if (allImageFiles.GetCount() > 0)
		for (i = 0; i < filePermutations; i++)	//随机交换次数
		{
			index1 = LMath::RandomInt(allImageFiles.GetCount());
			index2 = LMath::RandomInt(allImageFiles.GetCount());
			allImageFiles.Swap(index1, index2);
		}

	//将数据分成两部分，训练集和测试集
	printf("Splitting image files..\n");
	for (i = 0; i < allImageFiles.GetCount(); i++)
	{
		//打乱顺序后前一半作为训练集
		if (i < proportionTrain * allImageFiles.GetCount())
			trainImageFiles.Add(allImageFiles[i]);
		//后一半作为测试集
		else if
			(i < (proportionTrain + proportionTest) * allImageFiles.GetCount()) testImageFiles.Add(allImageFiles[i]);
	}
}

void LDataset::SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName)
{
	int points = labelImage.GetPoints();
	unsigned char *labelData = labelImage.GetData();
	for (int j = 0; j < points; j++, labelData++) (*labelData)++;
	labelImage.Save(fileName, domain);
}

int LDataset::Segmented(char *imageFileName)
{
	return(1);
}

void LDataset::GetLabelSet(unsigned char *labelset, char *imageFileName)
{
	memset(labelset, 0, classNo * sizeof(unsigned char));

	char *fileName;
	fileName = GetFileName(groundTruthFolder, imageFileName, groundTruthExtension);
	LLabelImage labelImage(fileName, this, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel);
	delete[] fileName;

	int points = labelImage.GetPoints(), i;
	unsigned char *labelData = labelImage.GetData();
	for (i = 0; i < points; i++, labelData++) if (*labelData) labelset[*labelData - 1] = 1;
}


LSowerbyDataset::LSowerbyDataset() : LDataset()
{
	seed = 10000;
	classNo = 7;
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;
	unaryWeighted = 0;

	proportionTrain = 0.5;
	proportionTest = 0.5;

	imageFolder = "Sowerby/Images/";
	imageExtension = ".png";
	groundTruthFolder = "Sowerby/GroundTruth/";
	groundTruthExtension = ".png";
	trainFolder = "Sowerby/Result/Train/";
	testFolder = "Sowerby/Result/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 150;
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 5;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Sowerby/Result/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 5;
	siftNumberOfClusters = 150;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "Sowerby/Result/Feature/Sift/";
	siftExtension = ".sft";

	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 5;
	colourSiftNumberOfClusters = 150;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Sowerby/Result/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 12;
	locationFolder = "Sowerby/Result/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Sowerby/Result/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 150;

	meanShiftXY[0] = 3.5;
	meanShiftLuv[0] = 3.5;
	meanShiftMinRegion[0] = 10;
	meanShiftFolder[0] = "Sowerby/Result/MeanShift/35x35/";
	meanShiftXY[1] = 5.5;
	meanShiftLuv[1] = 3.5;
	meanShiftMinRegion[1] = 10;
	meanShiftFolder[1] = "Sowerby/Result/MeanShift/55x35/";
	meanShiftXY[2] = 3.5;
	meanShiftLuv[2] = 5.5;
	meanShiftMinRegion[2] = 10;
	meanShiftFolder[2] = "Sowerby/Result/MeanShift/35x55/";

	meanShiftExtension = ".msh";

	denseNumRoundsBoosting = 500;
	denseBoostingSubSample = 5;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 2;
	denseMaximumRectangleSize = 100;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Sowerby/Result/Dense/";
	denseWeight = 1.0;
	denseMaxClassRatio = 0.2;

	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	cliqueMinLabelRatio = 0.5;
	cliqueThresholdRatio = 0.1;
	cliqueTruncation = 0.1;

	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 500;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.5;
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Sowerby/Result/Stats/";
	statsExtension = ".sts";

	Init();

	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);
	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);
	for (int i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i]);
	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);
}

void LSowerbyDataset::SetCRFStructure(LCrf *crf)
{
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(objDomain);

	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	crf->layers.Add(baseLayer);

	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];

	int i;
	for (i = 0; i < 3; i++)
	{
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));

	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);
}

LCorelDataset::LCorelDataset() : LDataset()
{
	seed = 10000;
	classNo = 7;	//7个类别
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	imageFolder = "Data/Corel/Images/";
	imageExtension = ".bmp";
	groundTruthFolder = "Data/Corel/GroundTruth/";
	groundTruthExtension = ".bmp";
	trainFolder = "Result/Corel/Train/";
	testFolder = "Result/Corel/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 5;	//5倍降采样
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/Corel/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 5;
	siftNumberOfClusters = 50;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "Result/Corel/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 5;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/Corel/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 12;	//位置数，归一化
	locationFolder = "Result/Corel/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/Corel/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 50;

	//第一层
	//搜索物理空间
	meanShiftXY[0] = 3.5;
	//颜色空间
	meanShiftLuv[0] = 3.5;
	meanShiftMinRegion[0] = 10;
	meanShiftFolder[0] = "Result/Corel/MeanShift/35x35/";
	//第二层
	meanShiftXY[1] = 5.5;
	meanShiftLuv[1] = 3.5;
	meanShiftMinRegion[1] = 10;
	meanShiftFolder[1] = "Result/Corel/MeanShift/55x35/";
	//第三层
	meanShiftXY[2] = 3.5;
	meanShiftLuv[2] = 5.5;
	meanShiftMinRegion[2] = 10;
	meanShiftFolder[2] = "Result/Corel/MeanShift/35x55/";

	meanShiftExtension = ".msh";

	//Boosting参数
	denseNumRoundsBoosting = 500;
	denseBoostingSubSample = 5;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 2;
	denseMaximumRectangleSize = 100;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Result/Corel/Dense/";
	//特征权重把
	denseWeight = 1.0;
	//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;

	//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 500;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Result/Corel/Stats/";
	statsExtension = ".sts";

	//共生信息训练路径
	cooccurenceTrainFile = "cooccurence.dat";
	//共生项权重
	cooccurenceWeight = 0.05;

	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//初始化
	Init();

	//创建目录
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);
	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);
	for (int i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i]);
	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);
}

void LCorelDataset::SetCRFStructure(LCrf *crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];

	int i;
	/*
	虽然有3个超像素层，但实际上模型只有两层，因为三个超像素都是建立在底层之上的
	for (i = 0; i < 3; i++)
	{
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}
	*/
//	/*
//	3 levels hierarchy (replace previous loop)
	//第一层的分割
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[0], meanShiftLuv[0], meanShiftMinRegion[0], meanShiftFolder[0], meanShiftExtension);
	//建立在底层上的分割
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);
	crf->layers.Add(superpixelLayer[0]);

	//2-3层的分割
	for (i = 1; i < 3; i++)
	{
		//不同尺度的聚类参数
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		//建立在第一层超像素层上的层
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, superpixelLayer[0], segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}
	//一致性先验项参数，不同懂
	consistencyPrior = 100000;
	//这两个高阶项应该就是Associative Hierarchical CRFs for Object Class Image Segmentation论文里的高阶数据项和平滑项
	//一致性高阶数据项
	LConsistencyUnarySegmentPotential *consistencyPotential = new LConsistencyUnarySegmentPotential(this, crf, objDomain, classNo, consistencyPrior);
	//基于第一个超像素层
	consistencyPotential->AddLayer(superpixelLayer[0]);
	crf->potentials.Add(consistencyPotential);
	//高阶平滑项，也是建立在第一个超像素层上
	crf->potentials.Add(new LHistogramPottsPairwiseSegmentPotential(this, crf, objDomain, superpixelLayer[0], classNo, pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta, pairwiseSegmentBuckets));
//	*/

	//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	/*
	//	random forest for unary pixel potential (replace previous 3 lines)
		const char *denseForestFile = "denseforest.dat";
		int denseTrees = 20;
		int denseDepth = 10;
		double denseForestDataRatio = 1.0;

		LRandomForest<int> *pixelForest = new LRandomForest<int>(trainFolder, "denseforest.dat", classNo, denseTrees, denseDepth, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int *, int, int))&LDenseUnaryPixelPotential::GetTrainForestValues, (int (LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetEvalForestValue, denseForestDataRatio);
		crf->learnings.Add(pixelForest);
		pixelPotential->learning = pixelForest;
	*/
	//一阶平滑项
	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));
	
	//高阶数据项，三个超像素层各一个
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);

	/*
	//	linear svm for unary segment potential (replace previous boosting lines)
		const char *statsLinSVMFile = "statslinsvm.dat";
		double statsLinLambda = 1e-1;
		int statsLinEpochs = 500;
		int statsLinSkip = 400;
		double statsLinBScale = 1e-2;

		LLinearSVM<double> *segmentLinSVM = new LLinearSVM<double>(trainFolder, statsLinSVMFile, classNo, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainSVMValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalSVMValues, statsLinLambda, statsLinEpochs, statsLinSkip, statsLinBScale);
		crf->learnings.Add(segmentLinSVM);
		statsPotential->learning = segmentLinSVM;
	*/

	/*
	//	approx kernel svm for unary segment potential (replace previous boosting lines)
		const char *statsApproxSVMFile = "statsapproxsvm.dat";
		double statsApproxLambda = 1e-5;
		int statsApproxEpochs = 500;
		int statsApproxSkip = 400;
		double statsApproxBScale = 1e-2;
		double statsApproxL = 0.0005;
		int statsApproxCount = 3;
		LApproxKernel<double> *kernel = new LIntersectionApproxKernel<double>();

		LApproxKernelSVM<double> *segmentApproxSVM = new LApproxKernelSVM<double>(trainFolder, statsApproxSVMFile, classNo, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainSVMValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalSVMValues, 1e-5, 500, 400, 1e-2, kernel, statsApproxL, statsApproxCount);
		crf->learnings.Add(segmentApproxSVM);
		statsPotential->learning = segmentApproxSVM;
	*/

	/*
	//	random forest for unary segment potential (replace previous boosting lines)
		const char *statsForestFile = "statsforest.dat";
		int statsTrees = 50;
		int statsDepth = 12;
		double statsForestDataRatio = 1.0;

		LRandomForest<double> *segmentForest = new LRandomForest<double>(trainFolder, statsForestFile, classNo, statsTrees, statsDepth, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalSVMValues, statsForestDataRatio);
		crf->learnings.Add(segmentForest);
		statsPotential->learning = segmentForest;
	*/

	//这个层暂时不清楚
	LPreferenceCrfLayer *preferenceLayer = new LPreferenceCrfLayer(crf, objDomain, this, baseLayer);
	crf->layers.Add(preferenceLayer);
	//共生平滑项，训练集中任意两个类别同时出现的概率
	crf->potentials.Add(new LCooccurencePairwiseImagePotential(this, crf, objDomain, preferenceLayer, trainFolder, cooccurenceTrainFile, classNo, cooccurenceWeight));
}

LMsrcDataset::LMsrcDataset() : LDataset()
{
	seed = 60000;
	featuresOnline = 0;
	unaryWeighted = 0;

	//类别数量
	classNo = 21;
	filePermutations = 10000;
	optimizeAverage = 1;

	//训练文件和测试文件
	trainFileList = "Data/Msrc/Train.txt";
	testFileList = "Data/Msrc/Test.txt";

	//训练数据比例
	proportionTrain = 0.5;
	//测试数据比例
	proportionTest = 0.5;

	//数据路径
	imageFolder = "Data/Msrc/Images/";
	imageExtension = ".bmp";
	groundTruthFolder = "Data/Msrc/GroundTruth/";
	groundTruthExtension = ".bmp";
	//保存路径
	trainFolder = "Result/Msrc/Train/";
	testFolder = "Result/Msrc/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	locationBuckets = 12;
	locationFolder = "Result/Msrc/Feature/Location/";
	locationExtension = ".loc";

	textonNumberOfClusters = 150;
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 5;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/Msrc/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 10;
	siftNumberOfClusters = 150;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7, siftSizes[3] = 9;
	siftSizeCount = 4;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "Result/Msrc/Feature/Sift/";
	siftExtension = ".sft";

	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 10;
	colourSiftNumberOfClusters = 150;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7, colourSiftSizes[3] = 9;
	colourSiftSizeCount = 4;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/Msrc/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/Msrc/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 150;

	//一阶数据项分类器迭代次数
	denseNumRoundsBoosting = 5000;
	//降采样
	denseBoostingSubSample = 5;
	denseNumberOfThetas = 25;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboostXX.dat";
	denseExtension = ".dns";
	denseFolder = "Result/Msrc/DenseXX/";
	denseWeight = 1.0;
	denseMaxClassRatio = 0.1;

	meanShiftXY[0] = 7.0;
	meanShiftLuv[0] = 6.5;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "Result/Msrc/MeanShift/70x65/";
	meanShiftXY[1] = 7.0;
	meanShiftLuv[1] = 9.5;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "Result/Msrc/MeanShift/70x95/";
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 14.5;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "Result/Msrc/MeanShift/70x145/";
	meanShiftExtension = ".msh";

	kMeansDistance[0] = 30;
	kMeansXyLuvRatio[0] = 1.0;
	kMeansFolder[0] = "Result/Msrc/KMeans/30/";
	kMeansDistance[1] = 40;
	kMeansXyLuvRatio[1] = 0.75;
	kMeansFolder[1] = "Result/Msrc/KMeans/40/";
	kMeansDistance[2] = 50;
	kMeansXyLuvRatio[2] = 0.6;
	kMeansFolder[2] = "Result/Msrc/KMeans/50/";
	kMeansIterations = 10;
	kMeansMaxDiff = 3;
	kMeansExtension = ".seg";

	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.6;
	pairwiseFactor = 6.4;
	pairwiseBeta = 16.0;

	cliqueMinLabelRatio = 0.5;
	cliqueThresholdRatio = 0.1;
	cliqueTruncation = 0.1;

	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	//高阶数据项分类器迭代次数
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.5;
	statsAlpha = 0.05;
	statsPrior = 0;
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Result/Msrc/Stats/";
	statsExtension = ".sts";

	//共生信息
	cooccurenceTrainFile = "cooccurence.dat";
	cooccurenceWeight = 0.006;

	//	Init();
	LDataset::Init();

	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);
	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i]);
	for (i = 0; i < 3; i++) ForceDirectory(kMeansFolder[i]);
	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);
}

void LMsrcDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);

	if ((label[0] == 5) || (label[0] == 8) || (label[0] == 19) || (label[0] == 20)) label[0] = 0;
	if (label[0] > 20) label[0] -= 4;
	else if (label[0] > 8) label[0] -= 2;
	else if (label[0] > 5) label[0]--;
}

void LMsrcDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	if (lab > 16) lab += 4;
	else if (lab > 6) lab += 2;
	else if (lab > 4) lab++;

	rgb[0] = rgb[1] = rgb[2] = 0;
	for (int i = 0; lab > 0; i++, lab >>= 3)
	{
		rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
		rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
		rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
	}
}

void LMsrcDataset::Init()
{
	FILE *f;
	char *fileName, name[20];

	f = fopen(trainFileList, "r");

	if (f != NULL)
	{
		fgets(name, 15, f);
		name[strlen(name) - 1] = 0;
		size_t size = strlen(name);
		while (size > 0)
		{
			fileName = new char[strlen(name) + 1];
			strncpy(fileName, name, strlen(name) - 4);
			fileName[strlen(name) - 4] = 0;

			trainImageFiles.Add(fileName);
			allImageFiles.Add(fileName);

			if (fgets(name, 15, f) == NULL) name[0] = 0;
			if (strlen(name) > 0) name[strlen(name) - 1] = 0;
			size = strlen(name);
		}
		fclose(f);
	}
	f = fopen(testFileList, "r");

	if (f != NULL)
	{
		fgets(name, 15, f);
		name[strlen(name) - 1] = 0;
		size_t size = strlen(name);
		while (size > 0)
		{
			fileName = new char[strlen(name) + 1];
			strncpy(fileName, name, strlen(name) - 4);
			fileName[strlen(name) - 4] = 0;
			testImageFiles.Add(fileName);
			allImageFiles.Add(fileName);

			if (fgets(name, 15, f) == NULL) name[0] = 0;
			if (strlen(name) > 0) name[strlen(name) - 1] = 0;
			size = strlen(name);
		}
		fclose(f);
	}
}

void LMsrcDataset::SetCRFStructure(LCrf *crf)
{
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(objDomain);

	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	crf->layers.Add(baseLayer);

	LPnCrfLayer *superpixelLayer[6];
	LSegmentation2D *segmentation[6];

	int i;

	for (i = 0; i < 3; i++)
	{
		segmentation[i] = new LKMeansSegmentation2D(kMeansXyLuvRatio[i], kMeansDistance[i], kMeansIterations, kMeansMaxDiff, kMeansFolder[i], kMeansExtension);
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}
	for (i = 0; i < 3; i++)
	{
		segmentation[3 + i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		superpixelLayer[3 + i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[3 + i], cliqueTruncation);
		crf->segmentations.Add(segmentation[3 + i]);
		crf->layers.Add(superpixelLayer[3 + i]);
	}
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));

	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	for (i = 0; i < 6; i++) statsPotential->AddLayer(superpixelLayer[i]);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);

	LPreferenceCrfLayer *preferenceLayer = new LPreferenceCrfLayer(crf, objDomain, this, baseLayer);
	crf->layers.Add(preferenceLayer);
	crf->potentials.Add(new LCooccurencePairwiseImagePotential(this, crf, objDomain, preferenceLayer, trainFolder, cooccurenceTrainFile, classNo, cooccurenceWeight));
}

LVOCDataset::LVOCDataset() : LDataset()
{
	seed = 10000;
	featuresOnline = 0;
	unaryWeighted = 0;

	classNo = 21;
	optimizeAverage = 1;

	imageFolder = "VOC2010/Images/";
	imageExtension = ".jpg";
	groundTruthFolder = "VOC2010/GroundTruth/";
	groundTruthExtension = ".png";

	trainFileList = "VOC2010/trainval.txt";
	testFileList = "VOC2010/test.txt";

	trainFolder = "VOC2010/Result/Train/";
	denseFolder = "VOC2010/Result/Dense/";
	statsFolder = "VOC2010/Result/Stats/";
	testFolder = "VOC2010/Result/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	locationBuckets = 20;
	locationFolder = "VOC2010/Result/Feature/Location/";
	locationExtension = ".loc";

	textonNumberOfClusters = 150;
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "VOC2010/Result/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 150;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9, siftSizes[3] = 11;
	siftSizeCount = 4;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "VOC2010/Result/Feature/Sift/";
	siftExtension = ".sft";

	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 150;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9, colourSiftSizes[3] = 11;
	colourSiftSizeCount = 4;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "VOC2010/Result/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "VOC2010/Result/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 150;

	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 15;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 10;
	denseMaximumRectangleSize = 250;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseWeight = 1.0;
	denseMaxClassRatio = 1.0;

	meanShiftXY[0] = 7.0;
	meanShiftLuv[0] = 7.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "VOC2010/Result/Segmentation/MeanShift70x70/";
	meanShiftXY[1] = 7.0;
	meanShiftLuv[1] = 10.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "VOC2010/Result/Segmentation/MeanShift70x100/";
	meanShiftXY[2] = 10.0;
	meanShiftLuv[2] = 7.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "VOC2010/Result/Segmentation/MeanShift100x70/";
	meanShiftXY[3] = 10.0;
	meanShiftLuv[3] = 10.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "VOC2010/Result/Segmentation/MeanShift100x100/";
	meanShiftExtension = ".seg";

	kMeansDistance[0] = 30;
	kMeansXyLuvRatio[0] = 1.0;
	kMeansFolder[0] = "VOC2010/Result/Segmentation/KMeans30/";
	kMeansDistance[1] = 40;
	kMeansXyLuvRatio[1] = 0.75;
	kMeansFolder[1] = "VOC2010/Result/Segmentation/KMeans40/";
	kMeansDistance[2] = 50;
	kMeansXyLuvRatio[2] = 0.6;
	kMeansFolder[2] = "VOC2010/Result/Segmentation/KMeans50/";
	kMeansDistance[3] = 60;
	kMeansXyLuvRatio[3] = 0.5;
	kMeansFolder[3] = "VOC2010/Result/Segmentation/KMeans60/";
	kMeansDistance[4] = 80;
	kMeansXyLuvRatio[4] = 0.375;
	kMeansFolder[4] = "VOC2010/Result/Segmentation/KMeans80/";
	kMeansDistance[5] = 100;
	kMeansXyLuvRatio[5] = 0.3;
	kMeansFolder[5] = "VOC2010/Result/Segmentation/KMeans100/";
	kMeansIterations = 5;
	kMeansMaxDiff = 2;
	kMeansExtension = ".seg";

	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;

	pairwisePrior = 3.0;
	pairwiseFactor = 12.0;
	pairwiseBeta = 16.0;

	cliqueMinLabelRatio = 0.5;
	cliqueThresholdRatio = 0.1;
	cliqueTruncation = 0.1;

	consistencyPrior = 0.05;

	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsPrior = 0;
	statsFactor = 0.3;
	statsMaxClassRatio = 1.0;
	statsAlpha = 0.05;
	statsTrainFile = "statsboost.dat";
	statsExtension = ".sts";

	cooccurenceTrainFile = "cooccurence.dat";
	cooccurenceWeight = 0.05;

	Init();

	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);
	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);
	for (i = 0; i < 4; i++) ForceDirectory(meanShiftFolder[i]);
	for (i = 0; i < 6; i++) ForceDirectory(kMeansFolder[i]);
	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);
}

void LVOCDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	if ((rgb[0] != 255) || (rgb[1] != 255) || (rgb[2] != 255))
	{
		for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);
		label[0]++;
	}
}

void LVOCDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	if (label[0] == 0) rgb[0] = rgb[1] = rgb[2] = 255;
	else
	{
		lab--;
		rgb[0] = rgb[1] = rgb[2] = 0;

		for (int i = 0; lab > 0; i++, lab >>= 3)
		{
			rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
			rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
			rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
		}
	}
}

void LVOCDataset::Init()
{
	FILE *f;
	char *fileName, name[12];

	f = fopen(trainFileList, "rb");

	if (f != NULL)
	{
		size_t size = fread(name, 1, 12, f);
		while (size == 12)
		{
			name[11] = 0;

			fileName = new char[strlen(name) + 1];
			strcpy(fileName, name);
			trainImageFiles.Add(fileName);
			allImageFiles.Add(fileName);

			size = fread(name, 1, 12, f);
		}
		fclose(f);
	}

	f = fopen(testFileList, "rb");

	if (f != NULL)
	{
		size_t size = fread(name, 1, 12, f);
		while (size == 12)
		{
			name[11] = 0;

			fileName = new char[strlen(name) + 1];
			strcpy(fileName, name);
			testImageFiles.Add(fileName);
			allImageFiles.Add(fileName);

			size = fread(name, 1, 12, f);
		}
		fclose(f);
	}
}

void LVOCDataset::SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName)
{
	labelImage.Save8bit(fileName);
}

void LVOCDataset::SetCRFStructure(LCrf *crf)
{
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(objDomain);

	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	crf->layers.Add(baseLayer);

	LPnCrfLayer *superpixelLayer[10];
	LSegmentation2D *segmentation[10];

	int i;
	for (i = 0; i < 6; i++)
	{
		segmentation[i] = new LKMeansSegmentation2D(kMeansXyLuvRatio[i], kMeansDistance[i], kMeansIterations, kMeansMaxDiff, kMeansFolder[i], meanShiftExtension);
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}
	for (i = 0; i < 4; i++)
	{
		segmentation[i + 6] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		superpixelLayer[i + 6] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i + 6], cliqueTruncation);
		crf->segmentations.Add(segmentation[i + 6]);
		crf->layers.Add(superpixelLayer[i + 6]);
	}

	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(coloursiftFeature);

	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);

	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));

	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio, 1);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	for (i = 0; i < 10; i++) statsPotential->AddLayer(superpixelLayer[i]);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;

	crf->potentials.Add(statsPotential);

	LPreferenceCrfLayer *preferenceLayer = new LPreferenceCrfLayer(crf, objDomain, this, baseLayer);
	crf->layers.Add(preferenceLayer);
	crf->potentials.Add(new LCooccurencePairwiseImagePotential(this, crf, objDomain, preferenceLayer, trainFolder, cooccurenceTrainFile, classNo, cooccurenceWeight));
}

LLeuvenDataset::LLeuvenDataset() : LDataset()
{
	seed = 60000;
	featuresOnline = 0;
	unaryWeighted = 0;

	//类别数
	classNo = 7;
	//样本随机参数
	filePermutations = 10000;
	optimizeAverage = 1;

	//数据集目录
	imageFolder = "Data/Leuven/Images/Left/";
	imageExtension = ".png";
	groundTruthFolder = "Data/Leuven/GroundTruth/";
	groundTruthExtension = ".png";
	trainFolder = "Result/Leuven/Train/";
	testFolder = "Result/Leuven/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	locationBuckets = 12;
	locationFolder = "Result/Leuven/Feature/Location/";
	locationExtension = ".loc";

	textonNumberOfClusters = 50;
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 5;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/Leuven/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 5;
	siftNumberOfClusters = 50;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7, siftSizes[3] = 9;
	siftSizeCount = 4;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "Result/Leuven/Feature/Sift/";
	siftExtension = ".sft";

	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 5;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7, colourSiftSizes[3] = 9;
	colourSiftSizeCount = 4;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/Leuven/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/Leuven/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 5;
	lbpNumberOfClusters = 50;

	//迭代次数
	denseNumRoundsBoosting = 500;
	denseBoostingSubSample = 5;
	denseNumberOfThetas = 25;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Result/Leuven/Dense/";
	denseWeight = 1.0;
	denseMaxClassRatio = 0.1;

	meanShiftXY[0] = 4.0;
	meanShiftLuv[0] = 2.0;
	meanShiftMinRegion[0] = 50;
	meanShiftFolder[0] = "Result/Leuven/MeanShift/40x20/";
	meanShiftXY[1] = 6.0;
	meanShiftLuv[1] = 3.0;
	meanShiftMinRegion[1] = 100;
	meanShiftFolder[1] = "Result/Leuven/MeanShift/60x30/";
	meanShiftXY[2] = 10.0;
	meanShiftLuv[2] = 3.5;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "Result/Leuven/MeanShift/100x35/";
	meanShiftExtension = ".msh";

	//模型的初始化参数
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	//主导标签最小比例？
	cliqueMinLabelRatio = 0.5;
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;

	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	//迭代次数
	statsNumberOfBoosts = 500;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.5;
	statsAlpha = 0.05;
	statsPrior = 0;
	//主导标签比例？
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Result/Leuven/Stats/";
	statsExtension = ".sts";

	//3D，有些参数不懂
	//测试结果保存路径
	dispTestFolder = "Result/Leuven/DispCrf/";
	//左目图像路径
	disparityLeftFolder = "Leuven/Images/Left/";
	//右目图像路径
	disparityRightFolder = "Leuven/Images/Right/";
	//深度GT图
	disparityGroundTruthFolder = "Leuven/DepthGroundTruth/";
	disparityGroundTruthExtension = ".png";
	disparityUnaryFactor = 0.1;
	disparityRangeMoveSize = 0;

	//视差参数
	//高斯滤波参数
	disparityFilterSigma = 5.0;
	//最大视差
	disparityMaxDistance = 500;
	disparityDistanceBeta = 500;
	//降采样
	disparitySubSample = 1;
	disparityMaxDelta = 2;

	disparityPairwiseFactor = 0.00005;
	//平滑截断概率
	disparityPairwiseTruncation = 10.0;
	//视差类别,很大
	disparityClassNo = 100;

	//相机基线
	cameraBaseline = 150;
	//相机高度
	cameraHeight = 200;
	//焦距
	cameraFocalLength = 472.391;
	//视角？
	cameraAspectRatio = 0.8998;
	//什么偏置？
	cameraWidthOffset = -2.0;
	cameraHeightOffset = -3.0;

	//模型参数初始化
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = -1e-4;
	//最大最小高度
	crossMinHeight = 0;
	crossMaxHeight = 1000;
	//高度聚类
	crossHeightClusters = 80;
	crossThreshold = 1e-6;
	//这个是要训练的
	crossTrainFile = "height.dat";

	//初始化
	Init();

	//创建文件夹
	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder, "train/");
	ForceDirectory(textonFolder, "train/");
	ForceDirectory(siftFolder, "train/");
	ForceDirectory(colourSiftFolder, "train/");
	ForceDirectory(locationFolder, "train/");
	ForceDirectory(lbpFolder, "train/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "train/");
	ForceDirectory(denseFolder, "train/");
	ForceDirectory(statsFolder, "train/");
	ForceDirectory(dispTestFolder, "train/");
	ForceDirectory(testFolder, "test/");
	ForceDirectory(textonFolder, "test/");
	ForceDirectory(siftFolder, "test/");
	ForceDirectory(colourSiftFolder, "test/");
	ForceDirectory(locationFolder, "test/");
	ForceDirectory(lbpFolder, "test/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "test/");
	ForceDirectory(denseFolder, "test/");
	ForceDirectory(statsFolder, "test/");
	ForceDirectory(dispTestFolder, "test/");
}

void LLeuvenDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
	}
			if (nameList[i] != NULL) free(nameList[i]);
}
		if (nameList != NULL) free(nameList);
}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}

//设置训练集和测试集
void LLeuvenDataset::Init()
{
	AddFolder("train/", trainImageFiles);
	AddFolder("test/", testImageFiles);
}

void LLeuvenDataset::SetCRFStructure(LCrf *crf)
{
	//目标CRF模型
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(objDomain);

	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	crf->layers.Add(baseLayer);

	//超像素层3层
	LPnCrfLayer *superpixelLayer[3];
	//对应的3个分割
	LSegmentation2D *segmentation[3];

	int i;
	for (i = 0; i < 3; i++)
	{
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}

	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//基于像素的数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	//dense boosting
	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//基于段的数据项，即目标的高阶项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);

	//segment boosting
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);

	//视差CRF模型
	LCrfDomain *dispDomain = new LCrfDomain(crf, this, disparityClassNo, dispTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LLeuvenDataset::DisparityRgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LLeuvenDataset::DisparityLabelToRgb);
	crf->domains.Add(dispDomain);

	//视差底层
	LBaseCrfLayer *dispBaseLayer = new LBaseCrfLayer(crf, dispDomain, this, disparityRangeMoveSize);
	crf->layers.Add(dispBaseLayer);

	//视差数据项
	LDisparityUnaryPixelPotential *pixelDispPotential = new LDisparityUnaryPixelPotential(this, crf, dispDomain, dispBaseLayer, disparityClassNo, disparityUnaryFactor, disparityFilterSigma, disparityMaxDistance, disparityDistanceBeta, disparitySubSample, disparityMaxDelta, disparityLeftFolder, disparityRightFolder);
	crf->potentials.Add(pixelDispPotential);

	//连接两个模型的基于目标高度分布的连接项
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, dispDomain, dispBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, disparityClassNo, cameraBaseline, cameraHeight, cameraFocalLength, cameraAspectRatio, cameraWidthOffset, cameraHeightOffset, crossMinHeight, crossMaxHeight, crossHeightClusters, crossThreshold, disparitySubSample);
	crf->potentials.Add(heightPotential);

	//二元项中的连接项，惩罚不一致度，即惩罚一个变化一个不变化的情况
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, dispDomain, dispBaseLayer, classNo, disparityClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, disparityPairwiseFactor, disparityPairwiseTruncation, crossPairwiseWeight));
}

void LLeuvenDataset::DisparityRgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[0] == 255) label[0] = 0;
	else label[0] = rgb[0] + 1;
}

void LLeuvenDataset::DisparityLabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = label[0] - 1;
}

void LLeuvenDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);

	switch (label[0])
	{
	case 1:
		label[0] = 1; break;
	case 7:
		label[0] = 2; break;
	case 12:
		label[0] = 3; break;
	case 20:
		label[0] = 4; break;
	case 21:
		label[0] = 5; break;
	case 36:
		label[0] = 6; break;
	case 38:
		label[0] = 7; break;
	default:
		label[0] = 0; break;
	}
}

void LLeuvenDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	switch (lab)
	{
	case 0:
		lab = 0; break;
	case 1:
		lab = 1; break;
	case 2:
		lab = 7; break;
	case 3:
		lab = 12; break;
	case 4:
		lab = 20; break;
	case 5:
		lab = 21; break;
	case 6:
		lab = 36; break;
	case 7:
		lab = 38; break;
	default:
		lab = 0;
	}
	rgb[0] = rgb[1] = rgb[2] = 0;
	for (int i = 0; lab > 0; i++, lab >>= 3)
	{
		rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
		rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
		rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
	}
}

LCamVidDataset::LCamVidDataset() : LDataset()
{
	seed = 60000;
	featuresOnline = 0;
	unaryWeighted = 0;

	classNo = 11;
	filePermutations = 10000;
	optimizeAverage = 1;

	//图像文件夹
	imageFolder = "Data/CamVid/Images/";
	//扩展名
	imageExtension = ".png";
	//标签文件夹
	groundTruthFolder = "Data/CamVid/GroundTruth/";
	//扩展名
	groundTruthExtension = ".png";
	//训练文件夹
	trainFolder = "Result/CamVid/Train/";
	//测试文件夹
	testFolder = "Result/CamVid/Crf/";

	//K-mean聚类参数，K-means应该是用于特征聚类的
	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	//位置特征参数设置
	locationBuckets = 12;
	locationFolder = "Result/CamVid/Feature/Location/";
	locationExtension = ".loc";

	//纹理特征参数设置
	textonNumberOfClusters = 50;	//filter respond聚类中心个数
	textonFilterBankRescale = 0.7;	//滤波器的参数吧
	textonKMeansSubSample = 10;		//??
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/CamVid/Feature/Texton/";
	textonExtension = ".txn";

	//sift特征参数设置，HOG
	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 20;
	siftNumberOfClusters = 50;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7, siftSizes[3] = 9;
	siftSizeCount = 4;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "Result/CamVid/Feature/Sift/";
	siftExtension = ".sft";

	//color-sift特征参数设置
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 20;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7, colourSiftSizes[3] = 9;
	colourSiftSizeCount = 4;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/CamVid/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	//LBP特征参数设置
	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/CamVid/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 50;

	//AdaBoost参数设置
	denseNumRoundsBoosting = 500;		//迭代次数，应该就是弱分类器个数
	denseBoostingSubSample = 5;		//降采样
	denseNumberOfThetas = 25;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Result/CamVid/Dense/";
	denseWeight = 1.0;
	denseMaxClassRatio = 0.1;

	//mean-sift参数设置
	meanShiftXY[0] = 3.0;
	meanShiftLuv[0] = 0.3;
	meanShiftMinRegion[0] = 200;
	meanShiftFolder[0] = "Result/CamVid/MeanShift/30x03/";
	meanShiftXY[1] = 3.0;
	meanShiftLuv[1] = 0.6;
	meanShiftMinRegion[1] = 200;
	meanShiftFolder[1] = "Result/CamVid/MeanShift/30x06/";
	meanShiftXY[2] = 3.0;
	meanShiftLuv[2] = 0.9;
	meanShiftMinRegion[2] = 200;
	meanShiftFolder[2] = "Result/CamVid/MeanShift/30x09/";
	meanShiftExtension = ".msh";

	//平滑项参数设置
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	//团的一些参数
	cliqueMinLabelRatio = 0.5;	//应该是主导标签比例
	cliqueThresholdRatio = 0.1;
	cliqueTruncation = 0.1;		//截断阈值，文章中的alpha吧

	//这些暂时不太清楚
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.5;
	statsAlpha = 0.05;
	statsPrior = 0;
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Result/CamVid/Stats/";
	statsExtension = ".sts";

	Init();

	//创建文件夹
	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder, "Train/");
	ForceDirectory(textonFolder, "Train/");
	ForceDirectory(siftFolder, "Train/");
	ForceDirectory(colourSiftFolder, "Train/");
	ForceDirectory(locationFolder, "Train/");
	ForceDirectory(lbpFolder, "Train/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "Train/");
	ForceDirectory(denseFolder, "Train/");
	ForceDirectory(statsFolder, "Train/");
	ForceDirectory(testFolder, "Val/");
	ForceDirectory(textonFolder, "Val/");
	ForceDirectory(siftFolder, "Val/");
	ForceDirectory(colourSiftFolder, "Val/");
	ForceDirectory(locationFolder, "Val/");
	ForceDirectory(lbpFolder, "Val/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "Val/");
	ForceDirectory(denseFolder, "Val/");
	ForceDirectory(statsFolder, "Val/");
	ForceDirectory(testFolder, "Test/");
	ForceDirectory(textonFolder, "Test/");
	ForceDirectory(siftFolder, "Test/");
	ForceDirectory(colourSiftFolder, "Test/");
	ForceDirectory(locationFolder, "Test/");
	ForceDirectory(lbpFolder, "Test/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "Test/");
	ForceDirectory(denseFolder, "Test/");
	ForceDirectory(statsFolder, "Test/");
}

void LCamVidDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);

	switch (label[0])
	{
	case 1:
		label[0] = 1; break;
	case 3:
		label[0] = 2; break;
	case 7:
		label[0] = 3; break;
	case 12:
		label[0] = 4; break;
	case 13:
		label[0] = 3; break;
	case 15:
		label[0] = 5; break;
	case 21:
		label[0] = 6; break;
	case 24:
		label[0] = 7; break;
	case 26:
		label[0] = 8; break;
	case 27:
		label[0] = 2; break;
	case 28:
		label[0] = 8; break;
	case 30:
		label[0] = 10; break;
	case 31:
		label[0] = 9; break;
	case 34:
		label[0] = 3; break;
	case 35:
		label[0] = 5; break;
	case 36:
		label[0] = 10; break;
	case 37:
		label[0] = 6; break;
	case 38:
		label[0] = 11; break;
	case 39:
		label[0] = 6; break;
	case 40:
		label[0] = 3; break;
	case 41:
		label[0] = 6; break;
	case 45:
		label[0] = 11; break;
	case 46:
		label[0] = 4; break;
	case 47:
		label[0] = 4; break;
	case 48:
		label[0] = 5; break;
	default:
		label[0] = 0; break;
	}
}

void LCamVidDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	switch (lab)
	{
	case 1:
		lab = 1; break;
	case 2:
		lab = 3; break;
	case 3:
		lab = 7; break;
	case 4:
		lab = 12; break;
	case 5:
		lab = 15; break;
	case 6:
		lab = 21; break;
	case 7:
		lab = 24; break;
	case 8:
		lab = 28; break;
	case 9:
		lab = 31; break;
	case 10:
		lab = 36; break;
	case 11:
		lab = 38; break;
	default:
		lab = 0;
	}
	rgb[0] = rgb[1] = rgb[2] = 0;
	for (int i = 0; lab > 0; i++, lab >>= 3)
	{
		rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
		rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
		rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
	}
}

//把指定文件夹下面的文件添加到相应的list中
void LCamVidDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);	//同时添加到所有文件中
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
	}
			if (nameList[i] != NULL) free(nameList[i]);
}
		if (nameList != NULL) free(nameList);
}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}

//添加文件
void LCamVidDataset::Init()
{
	//训练集和验证集都当作训练集
	AddFolder("Train/", trainImageFiles);
	//	AddFolder("Val/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}

void LCamVidDataset::SetCRFStructure(LCrf *crf)
{
	//创建CFR主体，包含一些基本信息
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体信息到CRF模型
	crf->domains.Add(objDomain);
	//创建底层模型，即基于像素层CRF
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到CRF模型
	crf->layers.Add(baseLayer);

	//高阶层，即基于segment的高阶CRF
	LPnCrfLayer *superpixelLayer[3];
	//高阶层的超像素段
	LSegmentation2D *segmentation[3];

	//分别创建高阶层模型
	int i;
	for (i = 0; i < 3; i++)
	{
		//首先进行无监督segmentation，获取不同scale的segment，此处是mean-sift分割算法
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		//然后基于segment map创建高阶层
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		//添加个segment map到CRF模型
		crf->segmentations.Add(segmentation[i]);
		//添加高阶层到CRF模型
		crf->layers.Add(superpixelLayer[i]);
	}

	//创建提取各个特征类
	//TextonBoost特征
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	//位置特征
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	//Sift特征，实际上作者说明的HOG特征
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	//基于LUV颜色空间的Sift特征
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	//LBP特征，描述纹理的
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	//添加各特征到CRF模型
	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//基于Dense Feature，即像素特征的底层一元势函数
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	//Dense Feature所用到的4种特征
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	//添加数据项到CRF模型
	crf->potentials.Add(pixelPotential);
	//用于像素级分类的Boosting分类器
	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	//添加分类器到CRF模型
	crf->learnings.Add(pixelBoosting);
	//学习算法设置为Boosting
	pixelPotential->learning = pixelBoosting;

	//创建基于8领域的Potts模型的二元势函数
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//添加平滑项到CRF模型
	crf->potentials.Add(pairwisePotentianl);

	//基于segment的一元势函数
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	//segment所用到的特征，segment的特征是基于dense feature的归一化直方图，描述特征分布情况
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	//分别为每一个高阶层添加一元势函数
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);
	//用于segment分类的Boosting分类器
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	//添加分类器到CRF模型
	crf->learnings.Add(segmentBoosting);
	//高阶势函数的学习设置为Boosting
	statsPotential->learning = segmentBoosting;
	//添加高阶势函数到CRF模型
	crf->potentials.Add(statsPotential);
}



/////////////////////////////////////////////////////////////////////////////////////////////
//
//				KITTI	Dataet
//
/////////////////////////////////////////////////////////////////////////////////////////////

//KITTI数据集
LKITTIDataset::LKITTIDataset()
{
	seed = 10000;
	classNo = 3;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	imageFolder = "Data/KITTI/Images/";
	imageExtension = ".png";
	groundTruthFolder = "Data/KITTI/GroundTruth/";
	groundTruthExtension = ".png";
	trainFolder = "Result/KITTI/Train/";
	testFolder = "Result/KITTI/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 5;	//5倍降采样
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/KITTI/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 5;
	siftNumberOfClusters = 50;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "Result/KITTI/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 5;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/KITTI/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 12;	//位置数，归一化
	locationFolder = "Result/KITTI/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/KITTI/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 50;

	//第一层
	//搜索物理空间
	meanShiftXY[0] = 7.0;
	//颜色空间
	meanShiftLuv[0] = 7.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "Result/KITTI/MeanShift/70x70/";
	//第二层
	meanShiftXY[1] = 7.0;
	meanShiftLuv[1] = 10.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "Result/KITTI/MeanShift/70x10/";
	//第三层
	meanShiftXY[2] = 10.0;
	meanShiftLuv[2] = 10.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "Result/KITTI/MeanShift/10x10/";

	meanShiftExtension = ".msh";

	//Boosting参数
	denseNumRoundsBoosting = 500;
	denseBoostingSubSample = 5;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 2;
	denseMaximumRectangleSize = 100;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Result/KITTI/Dense/";
	//特征权重把
	denseWeight = 1.0;
	//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;

	//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 500;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Result/KITTI/Stats/";
	statsExtension = ".sts";

	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);
	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);
	for (int i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i]);
	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);
}
void LKITTIDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	if (rgb[0]==255)		//有效点
	{
		if (rgb[2]==255)
		{
			label[0] = 1;		//道路
		}
		else	
			label[0] = 2;		//非路
	}
	else	
		label[0] = 0;		//无效
	
}
void LKITTIDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0]==2)
	{
		rgb[0] = 255;
	}
	else
	{
		if (label[0]==1)
		{
			rgb[0] = 255, rgb[2] = 255;
		}
	}	
}
void LKITTIDataset::SetCRFStructure(LCrf *crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];

	
	//	3 levels hierarchy
	//第一层的分割
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[0], meanShiftLuv[0], meanShiftMinRegion[0], meanShiftFolder[0], meanShiftExtension);
	//建立在底层上的分割
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);
	crf->layers.Add(superpixelLayer[0]);

	//2-3层的分割
	int i;
	for (i = 1; i < 3; i++)
	{
		//不同尺度的聚类参数
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		//建立在第一层超像素层上的层
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, superpixelLayer[0], segmentation[i], cliqueTruncation);
		crf->segmentations.Add(segmentation[i]);
		crf->layers.Add(superpixelLayer[i]);
	}
	//一致性先验项参数，不同懂
	consistencyPrior = 100000;
	//这两个高阶项应该就是Associative Hierarchical CRFs for Object Class Image Segmentation论文里的高阶数据项和平滑项
	//一致性高阶数据项
	LConsistencyUnarySegmentPotential *consistencyPotential = new LConsistencyUnarySegmentPotential(this, crf, objDomain, classNo, consistencyPrior);
	//基于第一个超像素层
	consistencyPotential->AddLayer(superpixelLayer[0]);
	crf->potentials.Add(consistencyPotential);
	//高阶平滑项，也是建立在第一个超像素层上
	crf->potentials.Add(new LHistogramPottsPairwiseSegmentPotential(this, crf, objDomain, superpixelLayer[0], classNo, pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta, pairwiseSegmentBuckets));

	//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//一阶平滑项
	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));

	//高阶数据项，三个超像素层各一个
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);
	//这个层暂时不清楚
	LPreferenceCrfLayer *preferenceLayer = new LPreferenceCrfLayer(crf, objDomain, this, baseLayer);
	crf->layers.Add(preferenceLayer);
	//共生平滑项，训练集中任意两个类别同时出现的概率
//	crf->potentials.Add(new LCooccurencePairwiseImagePotential(this, crf, objDomain, preferenceLayer, trainFolder, cooccurenceTrainFile, classNo, cooccurenceWeight));

}

