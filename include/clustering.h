#ifndef __clustering
#define __clustering

#include <stdio.h>
#include "std.h"

//聚类父类
template <class T>
class LClustering
{
	protected :
		//聚类文件夹和文件
		const char *clusterFolder, *clusterFile;
		//通道数
		int bands;
	public :
		//构造函数
		LClustering(const char *setClusterFolder, const char *setClusterFile, int setBands);
		virtual ~LClustering() {};

		//最近邻居
		virtual int NearestNeighbour(T *values) = 0;
		//聚类
		virtual void Cluster(T *data, int size) = 0;
		//载入
		virtual void LoadTraining() = 0;
		virtual void LoadTraining(FILE *f) = 0;
		//保存
		virtual void SaveTraining() = 0;
		virtual void SaveTraining(FILE *f) = 0;
		//应该是获取聚类中心数
		virtual int GetClusters() = 0;
};

//栅格聚类
template <class T>
class LLatticeClustering : public LClustering<T>
{
	private :
		T *minValues, *maxValues;
		//应该是聚类中心点
		int *buckets;	
	public :
		LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int setBuckets);
		LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int *setBuckets);
		~LLatticeClustering();

		int NearestNeighbour(T *values);
		void Cluster(T *data, int size) {};
		void LoadTraining() {};
		void LoadTraining(FILE *f) {};
		void SaveTraining() {};
		void SaveTraining(FILE *f) {};
		int GetClusters();
};

//K-means聚类
template <class T>
class LKMeansClustering : public LClustering<T>
{
	private :
		//定义在类中的类，并没有采取继承
		class LCluster
		{
			private :
				int bands;
				double *means;		//均值
				double *variances;		//方差
			public :
				int count;
				double logMixCoeff, logDetCov;

				LCluster();
				LCluster(int setBands);
				~LCluster();

				void SetBands(int setBands);
				double *GetMeans();
				double *GetVariances();
		};
		//决策树节点
		class LKdTreeNode
		{
			public :
				int terminal;
				int *indices;
				double splitValue;
				int indiceSize, splitDim;
				LKdTreeNode *left, *right;	//左右孩子节点

				LKdTreeNode();
				LKdTreeNode(int *setIndices, int setIndiceSize);
				~LKdTreeNode();
				void SetAsNonTerminal(int setSplitDim, double setSplitValue, LKdTreeNode *setLeft, LKdTreeNode *setRight);
		};
		//决策树类
		class LKdTree
		{
			public :
				LKdTreeNode *root;

				LKdTree();
				LKdTree(T *data, int numberOfClusters, int bands, int pointsPerKDTreeCluster);
				~LKdTree();
				int NearestNeighbour(T *data, int bands, double *values, LKdTreeNode *node, double (*meassure)(double *, double *, int));
		};

		int numberOfClusters, pointsPerKDTreeCluster;
		double kMeansMaxChange;

		double *clusterMeans, *dataMeans, *dataVariances;
		LKdTree *kd;
		double (*meassure)(double *, double *, int);
		int finalClusters, normalize;

		double AssignLabels(T *data, int size, LCluster **tempClusters, int *labels);
	public :
		LKMeansClustering(const char *setClusterFolder, const char *setClusterFile, int setBands, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setNormalize = 1);
		~LKMeansClustering();

		//最近邻居
		int NearestNeighbour(T *values);
		//聚类
		void Cluster(T *data, int size);
		void FillMeans(double *data);
		void LoadTraining();
		void LoadTraining(FILE *f);
		void SaveTraining();
		void SaveTraining(FILE *f);
		int GetClusters();
		double *GetMeans();
};

#endif