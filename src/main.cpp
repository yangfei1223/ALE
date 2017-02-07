#include "../include/main.h"

int main(int argc, char *argv[])
{
	//il初始化
	ilInit();
	//读取主函数参数，from和to为起始帧数
	int from = 0, to = -1;
	if(argc == 3) from = atoi(argv[1]), to = atoi(argv[2]);
	if(argc == 4) from = atoi(argv[2]), to = atoi(argv[3]);

	//处理不同数据集
//	LDataset *dataset = new LMsrcDataset();
//	LDataset *dataset = new LVOCDataset();
//	LDataset *dataset = new LCamVidDataset();
//	LDataset *dataset = new LCorelDataset();
//	LDataset *dataset = new LSowerbyDataset();
//	LDataset *dataset = new LLeuvenDataset();

	LDataset *dataset = new LKITTIDataset();

	//创建CRF模型
	LCrf *crf = new LCrf(dataset);
	//用数据集构造CRF模型，包含特征，势函数等参数和网络结果的设置，添加各种对象
	dataset->SetCRFStructure(crf);

	//无监督分割,mean-sift聚类算法生成segment图，作为高阶层
//	crf->Segment(dataset->allImageFiles, from, to);
	//训练特征，利用训练集提取不同特征，生成不同聚类中心，生成聚类中心只使用全部的训练集
//	crf->TrainFeatures(dataset->trainImageFiles);	//训练集
	//特征评估，提取图像特征，分配每个特征向量到相应的聚类中心，从而降低特征维数
//	crf->EvaluateFeatures(dataset->allImageFiles, from, to);
	//训练势函数，一阶以及高阶数据项的训练，即分类器的训练
	crf->TrainPotentials(dataset->trainImageFiles);
	//评价，即测试，用分类器输出每个类别的概率
	crf->EvaluatePotentials(dataset->testImageFiles, from, to);
	//模型推断，alpha-expansion算法求解
	crf->Solve(dataset->testImageFiles, from, to);
	//结果评估统计
	crf->Confusion(dataset->testImageFiles, "results.txt");

	delete crf;
	delete dataset;

	return(0);

}


