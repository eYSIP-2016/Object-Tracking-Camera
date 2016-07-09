#ifndef IMAGE_H
#define IMAGE_H

#include <QThread>

class TLD;

class Image : public QThread
{
    Q_OBJECT
public:
    explicit Image(QObject *parent = 0);
    ~Image();

protected:
    TLD *tld;
    int runFlag;
    int mousePos[5];

protected:
    void run();

signals:
    void windowClosed();
};

#endif // IMAGE_H
