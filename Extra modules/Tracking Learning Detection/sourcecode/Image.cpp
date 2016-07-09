#include "Image.h"
#include "TLD/TLD.h"
#include "CameraDS.h"

using namespace cv;

void mouseCallback(int event, int x, int y, int flags, void *param)
{
    if( CV_EVENT_LBUTTONDOWN == event )
    {
        ((int*)param)[0] = x;
        ((int*)param)[1] = y;
    }
    else if( CV_EVENT_MOUSEMOVE == event && CV_EVENT_FLAG_LBUTTON == flags )
    {
        ((int*)param)[2] = x;
        ((int*)param)[3] = y;
        ((int*)param)[4] = 1;
    }
    else if( CV_EVENT_LBUTTONUP == event )
    {
        if( 1 == ((int*)param)[4] )
        {
            ((int*)param)[2] = x;
            ((int*)param)[3] = y;
            ((int*)param)[4] = 2;
        }
    }
}

Image::Image(QObject *parent) :
    QThread(parent)
{
    tld = 0;

    runFlag = 1;

    mousePos[4] = 0;

    namedWindow("Image");
    setMouseCallback("Image", mouseCallback, mousePos);

    start();
}

Image::~Image()
{
    wait();

    delete tld;
}

void Image::run()
{
    CameraDS cameraDS;

    if( cameraDS.OpenCamera(0, false, 640, 480) )
    {
        runFlag = 2;
    }
    else
    {
        runFlag = 0;

        return;
    }

    Mat frame;
    double multiple = 0.5;
    Mat lastGray;
    Mat currentGray;
    BoundingBox boundingBox;
    bool status;

    double start;
    double end;
    char fps[32];

    while( runFlag )
    {
        frame = cameraDS.QueryFrame();

        if( frame.empty() )
            break;

        flip(frame, frame, 1);

        if( 1 == mousePos[4] )
        {
            boundingBox.x = min(mousePos[0], mousePos[2]) - 2;
            boundingBox.y = min(mousePos[1], mousePos[3]) - 2;
            boundingBox.width = abs(mousePos[0] - mousePos[2]) + 4;
            boundingBox.height = abs(mousePos[1] - mousePos[3]) + 4;

            if( (boundingBox.width - 6) * multiple >= 15 && (boundingBox.height - 6) * multiple >= 15 )
            {
                rectangle(frame, boundingBox, Scalar(0, 255, 255), 3);
            }
            else
            {
                rectangle(frame, boundingBox, Scalar(0, 0, 255), 3);
            }
        }
        else if( 2 == mousePos[4] )
        {
            mousePos[4] = 0;

            boundingBox.width = (abs(mousePos[0] - mousePos[2]) - 2) * multiple;
            boundingBox.height = (abs(mousePos[1] - mousePos[3]) - 2) * multiple;

            if( boundingBox.width < 15 || boundingBox.height < 15 )
                continue;

            boundingBox.x = (min(mousePos[0], mousePos[2]) + 1) * multiple;
            boundingBox.y = (min(mousePos[1], mousePos[3]) + 1) * multiple;

            cvtColor(frame, lastGray, CV_BGR2GRAY);

            resize(lastGray, lastGray, Size(), multiple, multiple);

            if( tld )
            {
                delete tld;
            }
            tld = new TLD;
            tld->init(lastGray, boundingBox);

            status = true;

            continue;
        }

        if( tld )
        {
            cvtColor(frame, currentGray, CV_BGR2GRAY);

            resize(currentGray, currentGray, Size(), multiple, multiple);

            tld->processFrame(lastGray, currentGray, boundingBox, status);

            if( status )
            {
                boundingBox.x = boundingBox.x / multiple - 3;
                boundingBox.y = boundingBox.y / multiple - 3;
                boundingBox.width = boundingBox.width / multiple + 6;
                boundingBox.height = boundingBox.height / multiple + 6;

                rectangle(frame, boundingBox, Scalar(0, 255, 0), 3);
            }

            swap(lastGray, currentGray);
        }

        start = end;

        end = getTickCount();

        sprintf(fps, "fps : %0.2f", 1.0 / (end - start) * getTickFrequency());

        putText(frame,
                fps,
                Point(10, frame.size().height - 15),
                CV_FONT_HERSHEY_DUPLEX,
                0.5,
                Scalar(0, 255, 255));

        if( cvGetWindowHandle("Image") )
        {
            imshow("Image", frame);
        }
        else
        {
            runFlag = 0;

            emit windowClosed();
        }
    }
}
