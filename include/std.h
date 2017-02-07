#ifndef __std
#define __std

//多线程
//#define MULTITHREAD
#define MAXTHREAD 64

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#ifdef _WIN32
#ifdef _WIN64
#pragma comment(lib,"lib64\\DevIL.LIB")
#pragma comment(lib,"lib64\\ILU.LIB")
#pragma comment(lib,"lib64\\ILUT.LIB")
#else
#pragma comment(lib,"lib32\\DevIL.LIB")
#pragma comment(lib,"lib32\\ILU.LIB")
#pragma comment(lib,"lib32\\ILUT.LIB")
#endif
#endif

#ifndef _WIN32
class LCrfDomain;
class LDataset;
class LGreyImage;
class LRgbImage;
class LLuvImage;
class LLabImage;
class LLabelImage;
class LCostImage;
class LSegmentImage;
class LCrf;
class LCrfLayer;
class LBaseCrfLayer;
class LPnCrfLayer;
class LPreferenceCrfLayer;
#endif


//数学工具
namespace LMath
{
	static const double pi = (double)3.1415926535897932384626433832795;
	static const double positiveInfinity = (double)1e50;
	static const double negativeInfinity = (double)-1e50;
	static const double almostZero = (double)1e-12;

	void SetSeed(unsigned int seed);
	unsigned int RandomInt();
	unsigned int RandomInt(unsigned int maxval);
	unsigned int RandomInt(unsigned int minval, unsigned int maxval);
	double RandomReal();
	double RandomGaussian(double mi, double var);

	double SquareEuclidianDistance(double *v1, double *v2, int size);
	double KLDivergence(double *histogram, double *referenceHistogram, int size, double threshold);
	double GetAngle(double x, double y);
};

//链表模版
template <class T>
class LList
{
	private :
		int count, capacity;	//元素个数和容量
		T *items;	//元素指针

		//重设大小
		void Resize(int size);
		//快排
		void QuickSort(int from, int to, int (*sort)(T, T));
	public :
		LList();
		~LList();
		
		//迭代器
		T &operator[](int index);
		//添加
		T &Add(T value);
		//插入
		T &Insert(T value, int index);
		//删除
		void Delete(int index);
		//交换
		void Swap(int index1, int index2);
		//排序
		void Sort(int (*sort)(T, T));
		//返回元素指针
		T *GetArray();
		//返回元素个数
		int GetCount();
		//清空
		void Clear();
};

#ifdef _WIN32
#define thread_type HANDLE
#define thread_return DWORD
#define thread_defoutput 1
#else
#define thread_type pthread_t
#define thread_return void *
#define thread_defoutput NULL
#endif

#ifdef _WIN32
//错误码
void _error(char *str);
//获取文件名
char *GetFileName(const char *folder, const char *name, const char *extension);
//创建文件夹
void ForceDirectory(const char *dir);
void ForceDirectory(const char *dir, const char *subdir);
//获取处理器ID
int GetProcessors();
//初始化临界区
void InitializeCriticalSection();
//删除临界区
void DeleteCriticalSection();
//进入临界区
void EnterCriticalSection();
//退出临界区
void LeaveCriticalSection();
//创建新线程
thread_type NewThread(thread_return (*routine)(void *), void *param);
//线程完成
int ThreadFinished(thread_type thread);
//关闭线程
void CloseThread(thread_type *thread);
#endif

#endif