#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <opencv2/tracking.hpp>


using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace std::chrono;



#define SOURCE 0
const int LOW_RES = 2.5; // baisse la resolution pour opti
const float EXIGENCE = 0.8; 
const float F_FRAME = 60;    // frame / sec
const float F_DETECTION = 10;    // detection / sec
const float F_TRACKING = 5;    // tracking / sec
const int P_FRAME = 1000 / F_FRAME;        // nb millisec entre deux frames
const int P_DETECTION = 1000 / F_DETECTION; // nb millisec entre deux detections
const int P_TRACKING = 1000 / F_TRACKING;    // nb millisec entre deux tracages
const Scalar COLOR_RECT(0, 255, 0);



high_resolution_clock::time_point tf1, tf2, td1, td2, tt1, tt2;
duration<double, std::milli> dtf, dtd, dtt;
int tWaitf, tWaitd, tWaitt;



VideoCapture capture(SOURCE);
Mat frame, frameLowRes, frameGray;
int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
VideoWriter video("outcpp.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));



HOGDescriptor hog = HOGDescriptor();
vector<double> found_weights;
vector<Rect> locations;
vector<Rect2d> lastLocations;



MultiTracker trackers = MultiTracker();
vector<Ptr<Tracker>> algorithms;
vector<Rect2d> objects;
bool frameInit = false, updateTracker = false, stop = false;



// revoie une valeur arondit ex : round(14.743, 2) = 14.74
float round(float nb, int p) {
    return (float)((int)(nb * pow(10, p))) / pow(10, p);
}



void writeTxt(string txt, int x, int y) {
    cv::putText(frame, txt, cv::Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0));
}



void draw_locations(Mat& img, const vector< Rect2d >& locations, const Scalar& color)
{
    if (!locations.empty())
    {
        for (int i = 0; i < locations.size(); i++)
        {
            Rect2d loc = locations[i];
            rectangle(img, loc, color, 2);

            if (lastLocations.size() == locations.size()) {
                Rect2d lastLoc = lastLocations[i];



                float x1 = lastLoc.x + lastLoc.width / 2;
                float y1 = lastLoc.y + lastLoc.height / 2;
                float x2 = loc.x + loc.width / 2;
                float y2 = loc.y + loc.height / 2;

                line(img, Point(x1, y1), Point(x2, y2), color, 2);
            }
        }
    }
    lastLocations = locations;
}




void distance() {

    vector <Rect2d> a = trackers.getObjects();
    if (a.size() > 0) {

        for (int i = 0; i < a.size(); i++) {
            Mat cameraMatrix = (Mat_<float>(3, 3) << 605.57, 0., 334.90, 0., 591.46, 238.77, 0., 0., 1.);

            float Corner_top_left[2] = { float(a.at(i).x), float(a.at(i).y) };
            float Corner_top_right[2] = { float(a.at(i).x + a.at(i).width), float(a.at(i).y) };
            float Corner_bottom_left[2] = { float(a.at(i).x), float(a.at(i).y + a.at(i).height) };
            float Corner_bottom_right[2] = { float(a.at(i).x + a.at(i).width), float(a.at(i).y + a.at(i).height) };

            vector<Point2f> p2d;
            p2d.push_back(Point2f(Corner_top_left[0], Corner_top_left[1]));
            p2d.push_back(Point2f(Corner_top_right[0], Corner_top_right[1]));
            p2d.push_back(Point2f(Corner_bottom_right[0], Corner_bottom_right[1]));
            p2d.push_back(Point2f(Corner_bottom_left[0], Corner_bottom_left[1]));

            vector<Point3f> p3d;
            p3d.push_back(Point3f(0, 0, 0));
            p3d.push_back(Point3f(0, 60, 0));
            p3d.push_back(Point3f(175, 60, 0));
            p3d.push_back(Point3f(175, 0, 0));

            Vec3d rotationVec;
            Vec3d translationVec;


            try {
                solvePnP(p3d, p2d, cameraMatrix, Mat::zeros(1, 4, CV_64FC1), rotationVec, translationVec, false, SOLVEPNP_ITERATIVE);
                writeTxt("D : " + to_string(translationVec[2]), Corner_top_left[0] * LOW_RES, Corner_top_left[1] * LOW_RES - 10);
            }
            catch (std::exception& e) { cerr << "\n" << e.what() << endl; }
        }
    }
}






bool newFrame() {

    // met a jour la frame
    capture >> frame;
    if (frame.rows == 0 || frame.cols == 0) { stop = true; return false; }

    // baisse la resolution pour opti
    cv::resize(frame, frameLowRes, cv::Size(frame.cols / LOW_RES, frame.rows / LOW_RES));
    frameInit = true;

    draw_locations(frameLowRes, trackers.getObjects(), COLOR_RECT);

    cv::resize(frameLowRes, frame, cv::Size(frame.cols, frame.rows));

    distance();
    writeTxt("FPS : " + to_string(1000 / (dtf.count() + tWaitf)), 5, 20);
    writeTxt("DPS : " + to_string(1000 / (dtd.count() + tWaitd)), 5, 40);
    writeTxt("TPS : " + to_string(1000 / (dtt.count() + tWaitt)), 5, 60);
    video.write(frame);

    imshow("Video", frame); waitKey(1);
    return true;
}




// detecte les pietons sur la frame
bool newDetection() {

    if (stop) return false;
    if (frameInit) {

        cvtColor(frameLowRes, frameGray, COLOR_BGR2GRAY);
        hog.detectMultiScale(frameGray, locations, found_weights, 0, cv::Size(2, 2), cv::Size(10, 10), 1.02, 2.0);

        algorithms.clear();
        objects.clear();

        for (size_t i = 0; i < locations.size(); i++) {
            if (found_weights[i] > EXIGENCE) {
                algorithms.push_back(TrackerCSRT::create());
                objects.push_back(locations[i]);
            }
        }
        updateTracker = true;
    }
    return true;
}



// traque les pietons
bool newTracking() {

    if (stop) return false;
    if (frameInit) {

        // tracage des pietons
        if (trackers.getObjects().size() > 0) {
            trackers.update(frameLowRes);
        }

        // met à jour le tracker si une nouvelle detection à été effectuée
        if (updateTracker) {
            updateTracker = false;
            trackers = MultiTracker();
            trackers.add(algorithms, frameLowRes, objects);
            lastLocations.clear();
        }
    }
    return true;
}






// appelle newFrame() toute les P_FRAME millisecondes
void frameLoop() {

    tf1 = high_resolution_clock::now();

    if (newFrame()) {

        tf2 = high_resolution_clock::now();
        dtf = tf2 - tf1;

        tWaitf = max(0, (int)(P_FRAME - dtf.count()));
        std::this_thread::sleep_for(std::chrono::milliseconds(tWaitf));

        frameLoop();
    }
}
void detectionLoop() {

    td1 = high_resolution_clock::now();

    if (newDetection()) {

        td2 = high_resolution_clock::now();
        dtd = td2 - td1;

        tWaitd = max(0, (int)(P_DETECTION - dtd.count()));
        std::this_thread::sleep_for(std::chrono::milliseconds(tWaitd));

        detectionLoop();
    }
}
void trackingLoop() {

    tt1 = high_resolution_clock::now();

    if (newTracking()) {

        tt2 = high_resolution_clock::now();
        dtt = tt2 - tt1;

        tWaitt = max(0, (int)(P_TRACKING - dtt.count()));
        std::this_thread::sleep_for(std::chrono::milliseconds(tWaitt));

        trackingLoop();
    }
}



// lance les threads
int main() {
    hog.setSVMDetector(hog.getDefaultPeopleDetector());

    std::thread threadFrame(frameLoop);
    std::thread threadDetection(detectionLoop);
    std::thread threadTracking(trackingLoop);

    if (threadFrame.joinable())
        threadFrame.join();

    if (threadDetection.joinable())
        threadDetection.join();

    if (threadTracking.joinable())
        threadTracking.join();

    capture.release();
    video.release();
    destroyAllWindows();
}