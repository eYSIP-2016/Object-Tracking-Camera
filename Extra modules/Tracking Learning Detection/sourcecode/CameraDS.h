#ifndef CAMERA_H
#define CAMERA_H

#define WIN32_LEAN_AND_MEAN

#include <atlbase.h>

#pragma include_alias( "dxtrans.h", "qedit.h" )
#define __IDxtCompositor_INTERFACE_DEFINED__
#define __IDxtAlphaSetter_INTERFACE_DEFINED__
#define __IDxtJpeg_INTERFACE_DEFINED__
#define __IDxtKey_INTERFACE_DEFINED__

#include "qedit.h"
#include "dshow.h"

#include <windows.h>
#include "opencv2/core/core_c.h"

#define MYFREEMEDIATYPE(mt)	{if ((mt).cbFormat != 0)		\
					{CoTaskMemFree((PVOID)(mt).pbFormat);	\
					(mt).cbFormat = 0;						\
					(mt).pbFormat = NULL;					\
				}											\
				if ((mt).pUnk != NULL)						\
				{											\
					(mt).pUnk->Release();					\
					(mt).pUnk = NULL;						\
				}}									

class CameraDS  
{
private:

	bool m_bConnected, m_bLock, m_bChanged;

	int m_nWidth, m_nHeight;

	long m_nBufferSize;

	IplImage *m_pFrame;

	CComPtr<IGraphBuilder> m_pGraph;

	CComPtr<ISampleGrabber> m_pSampleGrabber;

	CComPtr<IMediaControl> m_pMediaControl;

	CComPtr<IMediaEvent> m_pMediaEvent;

	CComPtr<IBaseFilter> m_pSampleGrabberFilter;
	CComPtr<IBaseFilter> m_pDeviceFilter;
	CComPtr<IBaseFilter> m_pNullFilter;

	CComPtr<IPin> m_pGrabberInput;
	CComPtr<IPin> m_pGrabberOutput;
	CComPtr<IPin> m_pCameraOutput;
	CComPtr<IPin> m_pNullInputPin;

	bool BindFilter(int nCamIDX, IBaseFilter **pFilter);

	void SetCrossBar();

public:

	CameraDS();
	virtual ~CameraDS();

	//打开摄像头，nCamID指定打开哪个摄像头，取值可以为0,1,2,...
	//bDisplayProperties指示是否自动弹出摄像头属性页
	//nWidth和nHeight设置的摄像头的宽和高，如果摄像头不支持所设定的宽度和高度，则返回false
	bool OpenCamera(int nCamID, bool bDisplayProperties = true, int nWidth = 320, int nHeight = 240);

	//关闭摄像头，析构函数会自动调用这个函数
	void CloseCamera();

	//返回摄像头的数目
	//可以不用创建CameraDS实例，采用int c=CameraDS::CameraCount();得到结果。
	static int CameraCount(); 

	//根据摄像头的编号返回摄像头的名字
	//nCamID: 摄像头编号
	//sName: 用于存放摄像头名字的数组
	//nBufferSize: sName的大小
	//可以不用创建CameraDS实例，采用CameraDS::CameraName();得到结果。
	static int CameraName(int nCamID, char* sName, int nBufferSize);

	//返回图像宽度
	int GetWidth(){return m_nWidth;} 

	//返回图像高度
	int GetHeight(){return m_nHeight;}

	//抓取一帧，返回的IplImage不可手动释放！
	//返回图像数据的为RGB模式的Top-down(第一个字节为左上角像素)，即IplImage::origin=0(IPL_ORIGIN_TL)
	IplImage * QueryFrame();
};

#endif 
