#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/photo.hpp"
#include <math.h>
#include <iostream>
#include <QMessageBox>
#include <QInputDialog>
#include <fstream>
#include <time.h>
#include <QtCore/QCoreApplication>
#include <QMovie>
#include <algorithm>    // std::reverse
#include <vector>       // std::vector

using namespace cv;
using namespace std;

const double amin = 1000;
const double amax = 10000000000;
const float cosAngle = 0.3;

Mat thresh,image;
String imagePath;

float realAreaSquare;
bool checkBoxArea,checkBoxSumAreas,checkBoxWid,checkBoxLen,checkBoxWidLen,checkBoxAveDev,checkBoxPerimeter;

vector<vector<Point> > square;
vector<vector<Point> > leaves;
vector<vector<Point> > leavesPCA;
vector<double> leavesPer;
vector<double> leavesArea;

QString result = "",auxExport = "",auxExport2 = "";
QString removeLeaf,species,treatment,replicate;

static QSqlDatabase database;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow){
        ui->setupUi(this);

        QString dir = qApp->applicationDirPath();
        QString dirdatabase = dir+"/database/database.db";

        QString dirdriver = dir+"/sqldrivers/";
        QCoreApplication::addLibraryPath(dirdriver);

        database = QSqlDatabase::addDatabase("QSQLITE");
        database.setDatabaseName(dirdatabase);
        if(!database.open()){
            ui->displayHist->setText("Unable to connect to bank.");
            ui->scrollHist->setWidget(ui->displayHist);
            ui->scrollHist->setAlignment(Qt::AlignHCenter);
        }

        //Visible only after calculate the dimensions
        ui->labelResult->setVisible(0);
        ui->scrollResult->setVisible(0);
        ui->btnExport->setVisible(0);
        ui->btnRemove->setVisible(0);
    }

MainWindow::~MainWindow(){
    delete ui;
}

Point2f GetPointAfterRotate(Point2f inputpoint,Point2f center,double angle){
    Point2d preturn;
    preturn.x = (inputpoint.x - center.x)*cos(-angle) - (inputpoint.y - center.y)*sin(-angle)+center.x;
    preturn.y = (inputpoint.x - center.x)*sin(-angle) + (inputpoint.y - center.y)*cos(-angle)+center.y;
    return preturn;
}

/*Point GetPointAfterRotate(Point inputpoint,Point center,double angle){
    Point preturn;
    preturn.x = (inputpoint.x - center.x)*cos(-1*angle) - (inputpoint.y - center.y)*sin(-1*angle)+center.x;
    preturn.y = (inputpoint.x - center.x)*sin(-1*angle) + (inputpoint.y - center.y)*cos(-1*angle)+center.y;
    return preturn;
}*/

double getOrientation(vector<Point> &pts, Point2f& pos){
    //Construct a buffer used by the pca analysis
    Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i){
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);

    //Store the position of the object
    pos = Point2f(pca_analysis.mean.at<double>(0, 0),
    pca_analysis.mean.at<double>(0, 1));

    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i){
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
        pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i,0);
    }
    return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

static double cosineAngle( Point pt1, Point pt2, Point pt0 ){
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void pca(vector<vector<Point> >& contours, int i){
    Point2f* pos = new Point2f();
    double dOrient =  getOrientation(contours[i], *pos);
    /*int xmin = 99999;
    int xmax = 0;
    int ymin = 99999;
    int ymax = 0;*/

    for (size_t j = 0;j<contours[i].size();j++){
        contours[i][j] = GetPointAfterRotate(contours[i][j],(Point)*pos,dOrient);
        /*if (contours[i][j].x < xmin)
            xmin = contours[i][j].x;
        if (contours[i][j].x > xmax)
            xmax = contours[i][j].x;
        if (contours[i][j].y < ymin)
            ymin = contours[i][j].y;
        if (contours[i][j].y > ymax)
            ymax = contours[i][j].y;*/
     }
}

/*static void draw( Mat& image, const vector<vector<Point> >& contornos){
    for( size_t i = 0; i < contornos.size(); i++ ){
        const Point* p = &contornos[i][0];
        int n = (int)contornos[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,0, 255), 20, LINE_AA);
    }

    namedWindow("Window 3", WINDOW_NORMAL);

    imshow("Window 3",image);

    imwrite("contours.jpg", image);
}*/

static void findObjects(){
    Mat gray;
    vector<vector<Point> > contours;
    vector<Point> approx;

    cvtColor( image, gray, COLOR_BGR2GRAY );
    threshold( gray, thresh, 60, 255, THRESH_BINARY_INV|THRESH_OTSU);
    findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    //imwrite("threshold.jpg", thresh);

    //draw(image, contours);

    for(size_t i = 0; i < contours.size(); i++){
        double auxper = arcLength(contours[i], true);
        approxPolyDP(contours[i], approx, auxper*0.02, true);
        double auxarea = fabs(contourArea(contours[i]));

        if(approx.size() == 4 && auxarea > amin && auxarea < amax && isContourConvex(approx)){
            double maxCosine = 0;

            for(int j = 2; j < 5; j++){
                double cosine = fabs(cosineAngle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
             }

             if(maxCosine < cosAngle){
                 square.push_back(contours[i]);
             }
             else{
                 leaves.push_back(contours[i]); //to drawing contours
                 pca(contours,i);
                 leavesPCA.push_back(contours[i]); //to calculate
                 leavesPer.push_back(auxper);
                 leavesArea.push_back(auxarea);
             }
        }else if(auxarea > amin && auxarea < amax){
            leaves.push_back(contours[i]);
            pca(contours,i);
            leavesPCA.push_back(contours[i]);
            leavesPer.push_back(auxper);
            leavesArea.push_back(auxarea);
        }
    }
}

static void surfaceCalc(){
    result.clear();
    auxExport.clear();
    auxExport2.clear();

    //-----------------------Auxiliary variables (calculations)------------------------------------
    float pixelsWidSquare=0, pixelsLenSquare=0, sum=0.0;
    float aveWidth=0.0, avgLength=0.0, aveArea=0.0, avePerimeter=0.0; //average
    float stdWidth=0.0, stdLength=0.0, stdArea=0.0, stdPerimeter=0.0; //standard deviation
    int size = leaves.size();
    float Width[size], Length[size], Area[size], Perimeter[size], WidLen[size];
    float pixelsAreaSquare=0.0, realPerSquare=0.0, pixelsPerSquare=0.0;

    //-----------------------Auxiliary variables (database)----------------------------------------
    QSqlQuery query;
    QString name = QString::fromStdString(imagePath);
    name = name.split("/")[name.split("/").size()-1];
    char dateStr [9]; _strdate(dateStr); char timeStr [9]; _strtime(timeStr);
    QString id_Image = dateStr; id_Image+=" "; id_Image+= timeStr;

    //-------------------------------------------SQUARE--------------------------------------------
     pixelsPerSquare = arcLength(square[0],true);

    if(checkBoxWid || checkBoxLen){
        pixelsWidSquare = ( pixelsPerSquare)/4;
        pixelsLenSquare = pixelsWidSquare;
    }

    if(checkBoxArea) pixelsAreaSquare = contourArea(square[0]);

    if(checkBoxPerimeter) realPerSquare = sqrt(float(realAreaSquare))* 4;

    //Draw square
    const Point* p = &square[0][0];
    int n = (int)square[0].size();
    polylines(image, &p, &n, 1, true, Scalar(0,255,0), 10, LINE_AA);

    //-------------------------------------------LEAVES--------------------------------------------
    vector<Rect> boundRect(size);

    float realSideSquare = sqrt(float(realAreaSquare));

    for(int i = 0; i < size; i++ ){

        //Draw and number leaves
        const Point* p = &leaves[i][0];
        int n = (int)leaves[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,0,255), 10, LINE_AA);
        putText(image, to_string(i+1),leaves[i].at(leaves[i].capacity()/2), FONT_HERSHEY_DUPLEX, 4, Scalar(255,0,0), 12);

        result.append("\nLeaf: "); result.append(QString::number(i+1));
        result.append("\n\n");

        //-----------------------------------------Width and Length----------------------------------------
        if(checkBoxWid || checkBoxLen){
            boundRect[i] = boundingRect(leavesPCA[i]);

            //float aux = sqrt((pow((boundRect[i].tl().x - boundRect[i].tl().x),2)+pow((boundRect[i].br().y - boundRect[i].tl().y),2)));
            float aux = boundRect[i].width;
            aux = (aux * realSideSquare)/pixelsWidSquare;

            //float aux2 = sqrt((pow((boundRect[i].tl().x  - boundRect[i].br().x),2)+pow((boundRect[i].tl().y - boundRect[i].tl().y),2)));
            float aux2 = boundRect[i].height;
            aux2 = (aux2 * realSideSquare)/pixelsLenSquare;

            if(aux2 > aux){
                 if(checkBoxAveDev && checkBoxWid) aveWidth += aux;
                 if(checkBoxAveDev && checkBoxLen) avgLength += aux2;
                 if(checkBoxWid){
                     result.append("\nWidth: "); result.append(QString::number(aux));
                     Width[i] = aux;
                 }
                 if(checkBoxLen){
                     result.append("\nLength: "); result.append(QString::number(aux2));
                     Length[i] = aux2;
                 }

                 if(checkBoxWidLen){
                     result.append("\nWidth/Length: "); result.append(QString::number(aux/aux2));
                     WidLen[i] = aux/aux2;
                 }

            }else{
                 if(checkBoxAveDev && checkBoxWid) aveWidth += aux2;
                 if(checkBoxAveDev && checkBoxLen) avgLength += aux;
                 if(checkBoxWid){
                     result.append("\nWidth: "); result.append(QString::number(aux2));
                      Width[i] = aux2;
                 }
                 if(checkBoxLen){
                     result.append("\nLength: "); result.append(QString::number(aux));
                     Length[i] = aux;
                 }
                 if(checkBoxWidLen){
                     result.append("\nWidth/Length: "); result.append(QString::number(aux2/aux));
                     WidLen[i] = aux2/aux;
                 }
             }
        }

        //-----------------------------------------------Area----------------------------------------------
        if(checkBoxArea){
            float auxArea = ((leavesArea[i] * realAreaSquare)/pixelsAreaSquare);
            if(checkBoxSumAreas) sum += auxArea;
            result.append("\nArea: "); result.append(QString::number(auxArea));
            if(checkBoxAveDev && checkBoxArea) aveArea += auxArea;
            Area[i] = auxArea;
        }

        //--------------------------------------------Perimeter--------------------------------------------
        if(checkBoxPerimeter){
            float auxPer = ((leavesPer[i] * realPerSquare)/pixelsPerSquare);
            result.append("\nPerimeter: "); result.append(QString::number(auxPer));
            if(checkBoxAveDev && checkBoxPerimeter) avePerimeter += auxPer;
            Perimeter[i] = auxPer;
        }

        result.append("\n\n");

        //----------------------------------------------Export---------------------------------------------
        if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxArea && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(WidLen[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen && checkBoxArea && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxArea){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(WidLen[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(WidLen[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen && checkBoxWidLen){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(WidLen[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen && checkBoxArea){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxArea && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxLen){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxArea){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxLen && checkBoxArea && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxLen && checkBoxArea){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append("\n");
        }
        else if(checkBoxLen && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxLen){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Length[i])); auxExport2.append("\n");
        }
        else if(checkBoxWid){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Width[i])); auxExport2.append("\n");
        }
        else if(checkBoxArea && checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
        else if(checkBoxArea){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Area[i])); auxExport2.append("\n");
        }
        else if(checkBoxPerimeter){
            auxExport2.append(QString::number(i+1));  auxExport2.append(",");
            auxExport2.append(QString::number(Perimeter[i])); auxExport2.append("\n");
        }
    } 


    //--------------------------------------------Result sum areas--------------------------------------------
    if(checkBoxSumAreas){
        result.append("\nSum areas: "); result.append(QString::number(sum));
        result.append("\n\n");
    }

    //-------------------------------------Average and standard deviation-------------------------------------
    if(checkBoxAveDev){
        //Average
        if(checkBoxAveDev && checkBoxWid){
            aveWidth = aveWidth / size;
            result.append("\nAverage width: "); result.append(QString::number(aveWidth));
        }

        if(checkBoxAveDev && checkBoxLen){
            avgLength = avgLength / size;
            result.append("\nAverage length: "); result.append(QString::number(avgLength));
        }

        if(checkBoxAveDev && checkBoxArea){
            aveArea = aveArea / size;
            result.append("\nAverage area: "); result.append(QString::number(aveArea));
        }

        if(checkBoxAveDev && checkBoxPerimeter){
            avePerimeter = avePerimeter / size;
            result.append("\nAverage perimeter: "); result.append(QString::number(avePerimeter));
        }

         result.append("\n\n");

        //Standard deviation
        if(checkBoxAveDev && checkBoxWid){
            for(int i = 0; i < size; i++ ){
                stdWidth += pow(Width[i]-aveWidth,2);
            }
            stdWidth = sqrt(stdWidth / size);
            result.append("\nWidth deviation: "); result.append(QString::number(stdWidth));
        }

        if(checkBoxAveDev && checkBoxLen){
            for(int i = 0; i < size; i++ ){
                stdLength += pow(Length[i]-avgLength,2);
            }
            stdLength = sqrt(stdLength / size);
            result.append("\nlength deviation: "); result.append(QString::number(stdLength));
        }

        if(checkBoxAveDev && checkBoxArea){
            for(int i = 0; i < size; i++ ){
                stdArea += pow(Area[i]-aveArea,2);
            }
            stdArea = sqrt(stdArea / size);
            result.append("\nArea deviation: "); result.append(QString::number(stdArea));
        }

        if(checkBoxAveDev && checkBoxPerimeter){
            for(int i = 0; i < size; i++ ){
                stdPerimeter += pow(Perimeter[i]-avePerimeter,2);
            }
            stdPerimeter = sqrt(stdPerimeter / size);
            result.append("\nPerimeter deviation: "); result.append(QString::number(stdPerimeter));
        }

         result.append("\n\n");
     }

    //----------------------Insert average and standard deviation in database and export----------------------
    if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxLen && checkBoxSumAreas && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Length,std_Length,ave_Area,std_Area,sum_Areas,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+
                      QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +","+QString::number(aveArea)+","+QString::number(stdArea)+","+QString::number(sum)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");

        auxExport2.append("Sum:,,,");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxLen && checkBoxSumAreas){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Length,std_Length,ave_Area,std_Area,sum_Areas) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+
                      QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +","+QString::number(aveArea)+","+QString::number(stdArea)+","+QString::number(sum)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");

        auxExport2.append("Sum:,,,");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxSumAreas && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Area,std_Area,sum_Areas,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(aveArea)+","+QString::number(stdArea)+","+QString::number(sum)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");

        auxExport2.append("Sum:,,"); auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxSumAreas){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,ave_Area,std_Area,sum_Areas) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(aveArea)+","+QString::number(stdArea)+","+QString::number(sum)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");

        auxExport2.append("Sum:,,"); auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxLen && checkBoxSumAreas && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,"
                      "ave_Length,std_Length,ave_Area,std_Area,sum_Areas,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +","+QString::number(aveArea)+","+QString::number(stdArea)+","+QString::number(sum)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");

        auxExport2.append("Sum:,,"); auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxLen && checkBoxSumAreas){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,"
                      "ave_Length,std_Length,ave_Area,std_Area,sum_Areas) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +","+QString::number(aveArea)+","+QString::number(stdArea)+","+QString::number(sum)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");

        auxExport2.append("Sum:,,"); auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxLen && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Length,std_Length,ave_Area,std_Area,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +","+QString::number(aveArea)+","+QString::number(stdArea)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxLen){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Length,std_Length,ave_Area,std_Area) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +","+QString::number(aveArea)+","+QString::number(stdArea)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxWid && checkBoxLen && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Length,std_Length,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(avgLength)+","+QString::number(stdLength)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        if(checkBoxWidLen) auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxWid && checkBoxLen){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Length,std_Length) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(avgLength)+","+QString::number(stdLength)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avgLength));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdLength));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxSumAreas && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Area,std_Area,sum_Areas,"
                      "ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveArea)
                      +","+QString::number(stdArea)+","+QString::number(sum)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");

        auxExport2.append("Sum:,"); auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxSumAreas){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Area,std_Area,sum_Areas) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveArea)
                      +","+QString::number(stdArea)+","+QString::number(sum)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");

        auxExport2.append("Sum:,"); auxExport2.append(QString::number(sum));
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Area,std_Area,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(aveArea)+","+QString::number(stdArea)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxWid){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width,"
                      "ave_Area,std_Area) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+","+QString::number(aveArea)+","+QString::number(stdArea)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxLen && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,"
                      "ave_Length,std_Length,ave_Area,std_Area,ave_Perimeter,std_Perimeter) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(avgLength)
                      +","+QString::number(stdLength)+","+QString::number(aveArea)+","+QString::number(stdArea)+","
                      +QString::number(avePerimeter)+","+QString::number(stdPerimeter)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxLen){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,"
                      "ave_Length,std_Length,ave_Area,std_Area) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(avgLength)
                      +","+QString::number(stdLength)+","+QString::number(aveArea)+","+QString::number(stdArea)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,"
                      "area_Square,ave_Area,std_Area,ave_Perimeter,std_Perimeter) values "
                      "('"+id_Image+"','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+
                      QString::number(realAreaSquare)+","+QString::number(aveArea)+","+
                      QString::number(stdArea)+","+QString::number(avePerimeter)+","+QString::number(stdPerimeter)
                      +")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveArea));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdArea)); auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxArea){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Area,std_Area) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveArea)+","+QString::number(stdArea)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveArea));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdArea)); auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxWid && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,"
                      "area_Square,ave_Width,std_Width, ave_Perimeter,std_Perimeter) values "
                      "('"+id_Image+"','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+
                      QString::number(realAreaSquare)+","+QString::number(aveWidth)+","+
                      QString::number(stdWidth)+","+QString::number(avePerimeter)+","+QString::number(stdPerimeter)
                      +")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxWid){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Width,std_Width) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(aveWidth)
                      +","+QString::number(stdWidth)+")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(aveWidth));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdWidth));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxLen && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,"
                      "area_Square,ave_Length,std_Length, ave_Perimeter,std_Perimeter) values "
                      "('"+id_Image+"','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+
                      QString::number(realAreaSquare)+","+QString::number(avgLength)+","+
                      QString::number(stdLength)+","+QString::number(avePerimeter)+","+QString::number(stdPerimeter)
                      +")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avgLength));auxExport2.append(",");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdLength));auxExport2.append(",");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxLen){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,ave_Length,std_Length) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(avgLength)+","+QString::number(stdLength)
                      +")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avgLength));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdLength));auxExport2.append("\n");
    }
    else if(checkBoxAveDev && checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,"
                      "area_Square, ave_Perimeter,std_Perimeter) values "
                      "('"+id_Image+"','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+
                      QString::number(realAreaSquare)+","+QString::number(avePerimeter)+","+QString::number(stdPerimeter)
                      +")");
        query.exec();

        auxExport2.append("Average:,");
        auxExport2.append(QString::number(avePerimeter));auxExport2.append("\n");

        auxExport2.append("Deviation:,");
        auxExport2.append(QString::number(stdPerimeter));auxExport2.append("\n");
    }
    else if(checkBoxArea && checkBoxSumAreas){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square,sum_Areas) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+","+QString::number(sum)+")");
        query.exec();

        if(checkBoxArea && checkBoxSumAreas && checkBoxWid && checkBoxLen && checkBoxWidLen){
            auxExport2.append("Sum:,,,,"); auxExport2.append(QString::number(sum));
        }
        else if(checkBoxArea && checkBoxSumAreas && checkBoxWid && checkBoxLen){
            auxExport2.append("Sum:,,,"); auxExport2.append(QString::number(sum));
        }
        else if(checkBoxArea && checkBoxSumAreas && checkBoxWid){
            auxExport2.append("Sum:,,"); auxExport2.append(QString::number(sum));
        }
        else if(checkBoxArea && checkBoxSumAreas && checkBoxLen){
            auxExport2.append("Sum:,,"); auxExport2.append(QString::number(sum));
        }
        else{
            auxExport2.append("Sum:,"); auxExport2.append(QString::number(sum));
        }


    }
    else if(checkBoxArea || checkBoxWid || checkBoxLen || checkBoxPerimeter){
        query.prepare("insert into image(id_Image,name,species,treatment,replicate,area_Square) values ('"+id_Image+
                      "','"+name+"','"+species+"','"+treatment+"','"+replicate+"',"+QString::number(realAreaSquare)+")");
        query.exec();
    }

    //--------------------------Insert Area, Width, Length and Perimeter in database--------------------------
    for(int i = 0; i < size; i++ ){
        if(checkBoxWid && checkBoxLen && checkBoxArea && checkBoxWidLen && checkBoxPerimeter){
                    query.prepare("insert into leaf(num_Leaf,id_Image,area,width,length,"
                                  "widthlength,perimeter) values ("+QString::number((i+1))+",'"+
                                  id_Image+"',"+QString::number(Area[i])+","+
                                  QString::number(Width[i])+","+QString::number(Length[i])+","+
                                  QString::number(WidLen[i])+","+QString::number(Perimeter[i])+")");
                    query.exec();
                }
        else if(checkBoxWid && checkBoxLen && checkBoxArea && checkBoxWidLen){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,width,length,widthlength) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Width[i])+","+QString::number(Length[i])+","+QString::number(WidLen[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxLen && checkBoxArea && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,width,length,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Width[i])+","+QString::number(Length[i])+","+
                          QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxLen && checkBoxArea){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,width,length) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Width[i])+","+QString::number(Length[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxArea && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,width,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Width[i])+","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxArea){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,width) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Width[i])+")");
            query.exec();
        }
        else if(checkBoxLen && checkBoxArea && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,length,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Length[i])+","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxLen && checkBoxArea){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,length) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Length[i])+")");
            query.exec();
        }
        else if(checkBoxArea && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,area,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxArea){
            query.prepare("insert into leaf(num_Leaf,id_Image,area) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Area[i])
                          +")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,width,length,widthlength,perimeter)"
                          " values ("+QString::number((i+1))+",'"+id_Image+"',"+
                          QString::number(Width[i])+","+QString::number(Length[i])+","+
                          QString::number(WidLen[i])+","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxLen && checkBoxWidLen){
            query.prepare("insert into leaf(num_Leaf,id_Image,width,length,widthlength) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Width[i])+","
                          +QString::number(Length[i])+","+QString::number(WidLen[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxLen && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,width,length,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Width[i])+","
                          +QString::number(Length[i])+","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxLen){
            query.prepare("insert into leaf(num_Leaf,id_Image,width,length) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Width[i])+","
                          +QString::number(Length[i])+")");
            query.exec();
        }
        else if(checkBoxWid && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,width,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+
                          QString::number( Width[i])+","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxWid){
            query.prepare("insert into leaf(num_Leaf,id_Image,width) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Width[i])+")");
            query.exec();
        }
        else if(checkBoxLen && checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,length,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Length[i])
                          +","+QString::number(Perimeter[i])+")");
            query.exec();
        }
        else if(checkBoxLen){
            query.prepare("insert into leaf(num_Leaf,id_Image,length) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Length[i])+")");
            query.exec();
        }
        else if(checkBoxPerimeter){
            query.prepare("insert into leaf(num_Leaf,id_Image,perimeter) values ("
                          +QString::number((i+1))+",'"+id_Image+"',"+QString::number(Perimeter[i])+")");
            query.exec();
        }
    }

    //-------------------------------------------------Export-------------------------------------------------
    auxExport.append("Image:," + name);
    auxExport.append("\nSpecies:," + species);
    auxExport.append("\nTreatment:," + treatment);
    auxExport.append("\nReplicate:," + replicate);
    auxExport.append("\nScale pattern area:,"); auxExport.append(QString::number(realAreaSquare));
    auxExport.append("\nNumber of leaves:,"); auxExport.append(QString::number(size));
    auxExport.append("\n\n");

    if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxArea && checkBoxPerimeter) auxExport.append("Number of leaf,Width,Length,Width/Length,Area,Perimeter\n\n");
    else if(checkBoxWid && checkBoxLen && checkBoxArea && checkBoxPerimeter) auxExport.append("Number of leaf,Width,Length,Area,Perimeter\n\n");
    else if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxArea) auxExport.append("Number of leaf,Width,Length,Width/Length,Area\n\n");
    else if(checkBoxWid && checkBoxLen && checkBoxWidLen && checkBoxPerimeter) auxExport.append("Number of leaf,Width,Length,Width/Length,Perimeter\n\n");
    else if(checkBoxWid && checkBoxLen && checkBoxWidLen) auxExport.append("Number of leaf,Width,Length,Width/Length\n\n");
    else if(checkBoxWid && checkBoxLen && checkBoxArea) auxExport.append("Number of leaf,Width,Length,Area\n\n");
    else if(checkBoxWid && checkBoxLen && checkBoxPerimeter) auxExport.append("Number of leaf,Width,Length,Perimeter\n\n");
    else if(checkBoxWid && checkBoxLen) auxExport.append("Number of leaf,Width,Length\n\n");
    else if(checkBoxWid && checkBoxArea && checkBoxPerimeter) auxExport.append("Number of leaf,Width,Area,Perimeter\n\n");
    else if(checkBoxWid && checkBoxArea) auxExport.append("Number of leaf,Width,Area\n\n");
    else if(checkBoxWid && checkBoxPerimeter) auxExport.append("Number of leaf,Width,Perimeter\n\n");
    else if(checkBoxLen && checkBoxArea && checkBoxPerimeter) auxExport.append("Number of leaf,Length,Area,Perimeter\n\n");
    else if(checkBoxLen && checkBoxArea) auxExport.append("Number of leaf,Length,Area\n\n");
    else if(checkBoxLen && checkBoxPerimeter) auxExport.append("Number of leaf,Length,Perimeter\n\n");
    else if(checkBoxLen) auxExport.append("Number of leaf,Length\n\n");
    else if(checkBoxWid) auxExport.append("Number of leaf,Width\n\n");
    else if(checkBoxArea && checkBoxPerimeter) auxExport.append("Number of leaf,Area,Perimeter\n\n");
    else if(checkBoxArea) auxExport.append("Number of leaf,Area\n\n");
    else if(checkBoxPerimeter) auxExport.append("Number of leaf,Perimeter\n\n");
}

void MainWindow::on_btnCalc_clicked(){
    if(!database.isOpen()){
        QMessageBox::warning(this, tr("Alert"),tr("Unable to connect to bank, the test will not come to bank."));
    }

    square.clear();
    leavesPCA.clear();
    leaves.clear();
    leavesPer.clear();
    leavesArea.clear();
    removeLeaf.clear();

    //Check boxes
    checkBoxArea = ui->checkBoxArea->isChecked();
    checkBoxSumAreas = ui->checkBoxSumAreas->isChecked();
    checkBoxWid = ui->checkBoxWid->isChecked();
    checkBoxLen = ui->checkBoxLen->isChecked();
    checkBoxWidLen = ui->checkBoxWidLen->isChecked();
    checkBoxAveDev = ui->checkBoxAveDev->isChecked();
    checkBoxPerimeter = ui->checkBoxPerimeter->isChecked();

    if(checkBoxWid || checkBoxLen  || checkBoxArea || checkBoxPerimeter){
        if(checkBoxWidLen && !(checkBoxWid && checkBoxLen)){
            QMessageBox::warning(this, tr("Alert"),tr("To calculate 'Width/Length' select the 'Width' and 'Length' options as well."));
        }
        else if(checkBoxSumAreas && !checkBoxArea){
            QMessageBox::warning(this, tr("Alert"),tr("To calculate 'Sum areas' select the 'Area' option as well."));
        }
        else if(ui->pathImage->text().toStdString() == "The image path will come here"){
            QMessageBox::warning(this, tr("Alert"),tr("Select an image."));
        }
        else{
            realAreaSquare = ui->areaSquare->text().toFloat();
            species = ui->species->text();
            treatment = ui->treatment->text();
            replicate = ui->replicate->text();

            image = imread(imagePath, IMREAD_COLOR );

            findObjects();
            surfaceCalc();

            QImage imageaux= QImage((uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
            ui->displayImage->setPixmap(QPixmap:: fromImage(imageaux).scaled((451*imageaux.width())/imageaux.height(),451));
            ui->labelResult->setVisible(1);
            ui->scrollResult->setVisible(1);
            ui->btnExport->setVisible(1);
            ui->btnRemove->setVisible(1);
            ui->displayResult->setText(result);
            ui->scrollResult->setWidget(ui->displayResult);
        }
    }
    else{
         QMessageBox::warning(this, tr("Alert"),tr("Check an option: 'Area', 'Width', 'length' or 'Perimeter'."));
    }
}

void MainWindow::on_btnExport_clicked(){
    QFileDialog dialog(this);
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName = QFileDialog::getSaveFileName(this,"");
    if(!fileName.isEmpty()){
        ofstream myfile;
        QString aux = fileName.toStdString().c_str();
        aux = aux.split(".csv")[0];
        aux+= ".csv";
        myfile.open(aux.toStdString());
        myfile.write((auxExport.toStdString().c_str()),auxExport.size());
        myfile.write((auxExport2.toStdString().c_str()),auxExport2.size());
        myfile.close();
    }
}

void MainWindow::on_btnClean_clicked(){
    ui->areaSquare->setText("1");
    ui->pathImage->setText("The image path will come here");
    ui->displayImage->setText("Select image will come here.");
    ui->labelResult->setVisible(0);
    ui->scrollResult->setVisible(0);
    ui->btnExport->setVisible(0);
    ui->btnRemove->setVisible(0);
    ui->species->setText("");
    ui->treatment->setText("");
    ui->replicate->setText("");
}

void MainWindow::on_btnRemove_clicked(){
    bool ok;
    removeLeaf = QInputDialog::getText(this, tr("Remove"),tr("Enter the contour number:"),
                                            QLineEdit::Normal,"1,2", &ok);
    if (ok && removeLeaf != ""){
        QSqlQuery query;
        if(query.exec("select max(id_Image) from image")){
            while(query.next()){
                QString auxIdimage = query.value(0).toString();
                query.prepare("delete from image where id_Image = '"+auxIdimage+"'");
                query.exec();
                query.prepare("delete from leaf where id_Image = '"+auxIdimage+"'");
                query.exec();
            }
        }

        for(int x = 0 ; x < removeLeaf.split(",").size(); x++){
            if(!leaves.empty()){
                if((removeLeaf.split(",")[x].toInt() - (x+1)) > -1 && (removeLeaf.split(",")[x].toInt() - (x+1)) < (int) leaves.size()){
                    leaves.erase(leaves.begin() + (removeLeaf.split(",")[x].toInt() - (x+1)));
                    leavesPCA.erase(leavesPCA.begin() + (removeLeaf.split(",")[x].toInt() - (x+1)));
                    leavesPer.erase(leavesPer.begin() + (removeLeaf.split(",")[x].toInt() - (x+1)));
                    leavesArea.erase(leavesArea.begin() + (removeLeaf.split(",")[x].toInt() - (x+1)));
                }else{
                    QMessageBox::warning(this, tr("Alert"),tr("An invalid number has been entered!"));
                }
            }else{
                QMessageBox::warning(this, tr("Alert"),tr("There aren't contours to remove."));
            }
        }

        image.release();
        image = imread(imagePath, IMREAD_COLOR );
        surfaceCalc();

        QImage imageaux= QImage((uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
        ui->displayImage->setPixmap(QPixmap:: fromImage(imageaux).scaled((451*imageaux.width())/imageaux.height(),451));
        ui->displayResult->setText(result);
        ui->scrollResult->setWidget(ui->displayResult);
    }
}

void MainWindow::on_tabWidget_currentChanged(){
    QSqlQuery query,query2;
    if(query.exec("select * from image order by id_Image desc")){
         QString aux = "";
         while(query.next()){
            aux += "\n" + query.value(0).toString() + "\n\n";
            aux += "Image: " + query.value(1).toString();
            aux += "\nSpecies: " + query.value(2).toString();
            aux += "\nTreatment: " + query.value(3).toString();
            aux += "\nReplicate: " + query.value(4).toString();
            aux += "\nScale pattern area: " + query.value(5).toString();
            aux += "\nAverage width: " + query.value(6).toString();
            aux += "\nWidth deviation: " + query.value(7).toString();
            aux += "\nAverage length: " + query.value(8).toString();
            aux += "\nlength deviation: " + query.value(9).toString();
            aux += "\nAverage area: " + query.value(10).toString();
            aux += "\nArea deviation: " + query.value(11).toString();
            aux += "\nSum areas: " + query.value(12).toString();
            aux += "\nAverage perimeter: " + query.value(13).toString();
            aux += "\nPerimeter deviation: " + query.value(14).toString()+ "\n\n";

            QString pathImage = query.value(0).toString();

            if(query2.exec("select * from leaf where id_Image = '" +pathImage+"'")){
                 while(query2.next()){
                    aux += "leaf: " + query2.value(2).toString() + "\n\n";
                    aux += "\nWidth: " + query2.value(3).toString();
                    aux += "\nlength: " + query2.value(4).toString();
                    aux += "\nWidth/Length: " + query2.value(5).toString();
                    aux += "\nArea: " + query2.value(6).toString();
                    aux += "\nPerimeter: " + query2.value(7).toString()+ "\n\n";
                }
                 aux+="________________________________________\n";
            }
        }

        if(aux == ""){
            ui->displayHist->setText("The historic will come here.");
        }else{
            ui->displayHist->setText(aux);
        }
        ui->scrollHist->setWidget(ui->displayHist);
        ui->scrollHist->setAlignment(Qt::AlignHCenter);
    }
}

void MainWindow::on_btnClearHistory_clicked(){
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Clear history", "This operation will delete the entire historic. Are you sure?", QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        QSqlQuery query;
        query.prepare("delete from leaf");
        query.exec();
        query.prepare("delete from image");
        query.exec();

        ui->displayHist->setText("The historic will come here.");
    }
}

void MainWindow::on_btnImage_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Images(*.* *.png *.jpeg *.jpg)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open Imagens"), "",tr("Image Files (*.* *.png *.jpeg *.jpg"));

    if(!fileName.isEmpty()){
        ui->pathImage->setText(fileName);
        imagePath = ui->pathImage->text().toStdString();
        image = imread(imagePath, IMREAD_COLOR );
        QImage imageaux= QImage((uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
        ui->displayImage->setPixmap(QPixmap:: fromImage(imageaux).scaled((451*imageaux.width())/imageaux.height(),451));
    }
}
