#include "../include/main.h"

int main(int argc, char *argv[])
{
	//il��ʼ��
	ilInit();
	//��ȡ������������from��toΪ��ʼ֡��
	int from = 0, to = -1;
	if(argc == 3) from = atoi(argv[1]), to = atoi(argv[2]);
	if(argc == 4) from = atoi(argv[2]), to = atoi(argv[3]);

	//����ͬ���ݼ�
//	LDataset *dataset = new LMsrcDataset();
//	LDataset *dataset = new LVOCDataset();
//	LDataset *dataset = new LCamVidDataset();
//	LDataset *dataset = new LCorelDataset();
//	LDataset *dataset = new LSowerbyDataset();
//	LDataset *dataset = new LLeuvenDataset();

	LDataset *dataset = new LKITTIDataset();

	//����CRFģ��
	LCrf *crf = new LCrf(dataset);
	//�����ݼ�����CRFģ�ͣ������������ƺ����Ȳ����������������ã���Ӹ��ֶ���
	dataset->SetCRFStructure(crf);

	//�޼ල�ָ�,mean-sift�����㷨����segmentͼ����Ϊ�߽ײ�
//	crf->Segment(dataset->allImageFiles, from, to);
	//ѵ������������ѵ������ȡ��ͬ���������ɲ�ͬ�������ģ����ɾ�������ֻʹ��ȫ����ѵ����
//	crf->TrainFeatures(dataset->trainImageFiles);	//ѵ����
	//������������ȡͼ������������ÿ��������������Ӧ�ľ������ģ��Ӷ���������ά��
//	crf->EvaluateFeatures(dataset->allImageFiles, from, to);
	//ѵ���ƺ�����һ���Լ��߽��������ѵ��������������ѵ��
	crf->TrainPotentials(dataset->trainImageFiles);
	//���ۣ������ԣ��÷��������ÿ�����ĸ���
	crf->EvaluatePotentials(dataset->testImageFiles, from, to);
	//ģ���ƶϣ�alpha-expansion�㷨���
	crf->Solve(dataset->testImageFiles, from, to);
	//�������ͳ��
	crf->Confusion(dataset->testImageFiles, "results.txt");

	delete crf;
	delete dataset;

	return(0);

}


