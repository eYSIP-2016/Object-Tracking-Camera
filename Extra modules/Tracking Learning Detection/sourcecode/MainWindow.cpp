#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "Image.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    image = new Image(this);

    connect(image, SIGNAL(windowClosed()), qApp, SLOT(quit()));
}

MainWindow::~MainWindow()
{
    delete image;

    delete ui;
}
