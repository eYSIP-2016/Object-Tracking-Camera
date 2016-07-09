#-------------------------------------------------
#
# Project created by QtCreator 2012-10-10T16:04:07
#
#-------------------------------------------------

QT += core gui

greaterThan(QT_MAJOR_VERSION, 4) : QT += widgets

TARGET = TLD
TEMPLATE = app

INCLUDEPATH += D:/OpenCV/2.4.5/build/include

CONFIG(debug, debug | release){
    LIBS += D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_core245d.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_highgui245d.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_imgproc245d.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_legacy245d.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_video245d.lib
}

CONFIG(release, debug | release){
    LIBS += D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_core245.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_highgui245.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_imgproc245.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_legacy245.lib \
        D:/OpenCV/2.4.5/build/x86/vc10/lib/opencv_video245.lib
}

SOURCES += main.cpp \
    MainWindow.cpp \
    Image.cpp \
    TLD/TLD.cpp \
    TLD/tld_utils.cpp \
    TLD/LKTracker.cpp \
    TLD/FerNNClassifier.cpp \
    CameraDS.cpp

HEADERS += MainWindow.h \
    Image.h \
    TLD/TLD.h \
    TLD/tld_utils.h \
    TLD/LKTracker.h \
    TLD/FerNNClassifier.h \
    CameraDS.h \
    qedit.h

FORMS += MainWindow.ui
