#ifndef __crf
#define __crf

#include "dataset.h"
#include "learning.h"
#include "graph.h"

class LCrfDomain;
class LCrf;
//CRF层父类
class LCrfLayer		
{
	protected :
		//目标类别数
		int classNo;
		LDataset *dataset;
	public :
		LCrf *crf;		
		LCrfDomain *domain;
		//父层，即上一层
		LCrfLayer *parent;		//类似与链表，每个结构中有一个指向下一个的指针
		//标签
		unsigned char *labels, *active;
		int nodeOffset;
		int range;
		//构造函数
		LCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, int setRange);
		//析构函数
		~LCrfLayer();

		//初始化
		virtual void Initialize(char *imageFileName, int onFly = 0) {};
		//设置标签
		virtual void SetLabels() {};
		//反初始化，应该类似于Release或者Dump
		virtual void UnInitialize() {};
		//获取顶点数
		virtual int GetNodeCount() = 0;
		//获取边数
		virtual int GetEdgeCount() = 0;
		//获取成对的节点数
		virtual int GetPairwiseNodeCount();
		//建立图模型
		virtual void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes) {};
		//更新标签
		virtual int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//???
		virtual int BinaryNodes();
};

//底层，即像素层
class LBaseCrfLayer : public LCrfLayer
{
	protected :
		int width, height;	//图像宽和高
	public :
		//构造函数
		LBaseCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, int setRange);
		//析构函数
		~LBaseCrfLayer();

		//初始化
		void Initialize(char *imageFileName, int onFly = 0);
		//反初始化
		void UnInitialize();
		//获取节点数
		int GetNodeCount();
		//获取边缘数
		int GetEdgeCount();
		//建立图模型
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//更新标签
		int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//???
		int BinaryNodes();
		//获取成对节点数
		int GetPairwiseNodeCount();
};

//Pn高阶层
class LPnCrfLayer : public LCrfLayer
{
	protected :
	public :
		int *segmentCounts, **segmentIndexes, segmentCount;
		int *baseSegmentCounts, **baseSegmentIndexes;
		LSegmentation2D *segmentation;
		double **weights, *weightSums, truncation;
		const char *segFolder, *segExtension;

		LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, LSegmentation2D *setSegmentation, double setTruncation);
		LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, const char *setSegFolder, const char *setSegExtension, double setTruncation);
		~LPnCrfLayer();

		void Initialize(char *imageFileName, int onFly = 0);
		void SetLabels();
		void UnInitialize();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int BinaryNodes();
		int GetPairwiseNodeCount();
};

//？优先层？
class LPreferenceCrfLayer : public LCrfLayer
{
	protected :
	public :
		LPreferenceCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent);
		~LPreferenceCrfLayer();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void Initialize(char *imageFileName, int onFly = 0);
		void UnInitialize();
};

//CRF主体
class LCrfDomain
{
	protected :
	public :
		LCrf *crf;		//指向模型的指针
		LDataset *dataset;		//指向数据集的指针
		LBaseCrfLayer *baseLayer;		//指向底层的指针
		int classNo;	//类编号
		const char *testFolder;		//测试文件夹
		//构造函数
		LCrfDomain(LCrf *setCrf, LDataset *setDataset, int setClassNo, const char *setTestFolder, void (LDataset::*setRgbToLabel)(unsigned char *, unsigned char *), void (LDataset::*setLabelToRgb)(unsigned char *, unsigned char *));
		//标签转换
		void (LDataset::*rgbToLabel)(unsigned char *, unsigned char *);
		void (LDataset::*labelToRgb)(unsigned char *, unsigned char *);
};

//CRF模型
class LCrf
{
	private :
		//数据集
		LDataset *dataset;
		//图模型
		Graph<double, double, double> *g;
		//图模型节点
		Graph<double, double, double>::node_id *nodes;

	public :
		//包含的主体个数,在联合模型中会用到，比如分割和3D重建的联合模型
		LList<LCrfDomain *> domains;
		//CRF层
		LList<LCrfLayer *> layers;
		//势函数
		LList<LPotential *> potentials;
		//特征
		LList<LFeature *> features;
		//高层无监督的分割
		LList<LSegmentation2D *> segmentations;
		//学习
		LList<LLearning *> learnings;
		//数据集
		LCrf(LDataset *setDataset);
		~LCrf();
		//分割原图
		void Segment(char *imageFileName);
		void Segment(LList<char *> &imageFiles, int from = 0, int to = -1);
		//训练特征
		void TrainFeatures(LList<char *> &imageFiles);
		//提取特征
		void EvaluateFeatures(LList<char *> &imageFiles, int from = 0, int to = -1);
		//训练势函数
		void TrainPotentials(LList<char *> &imageFiles);
		//势函数测试，主要是数据项的分类器对数据的测试
		void EvaluatePotentials(LList<char *> &imageFiles, int from = 0, int to = -1);
		//模型后处理优化求解
		void Solve(LList<char *> &imageFiles, int from = 0, int to = -1);

		//初始化和反初始化求解器
		void InitSolver(char *imageFileName, LLabelImage &labelImage);
		void UnInitSolver();
		void Solve(char *imageFileName);

		//alpha-expand算法
		int Expand(LCrfDomain *domain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//结果评估,评估测试结果
		void Confusion(LList<char *> &imageFiles, char *confusionFileName);
		void Confusion(LList<char *> &imageFiles, char *confusionFileName, int maxError);
};

#endif