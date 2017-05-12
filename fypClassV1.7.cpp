#include "opencv2/opencv.hpp"
//#include <iostream>
//Added support for detection on every other frame
class Driver {
public:
	Driver(void);
	cv::Mat Detect(cv::Mat);
	void BlobThold(int, int, int);
	void setParams(int, int, int, int, double);//thold1 thold2 alert step drowsiness minEye
	void flipParams(bool, bool, bool, bool, bool, bool,bool);//alertOn drowsyOn bwFrame noseDetect,mouthdetect
private:
	bool FindFace(cv::Mat);
	bool FindEye(bool);
	bool BlobEye(bool);//detect open or closed eye by using blob detector
	bool BlobNose();//true for head ahead false for looking aside
	bool BlobMouth();
	double BoundNum(double, double, double);
	unsigned char leftStats = 255, rightStats = 255, headStats, mouthStats = 0;//save last 8 eye states in binary
	double fixedLeftEyePos[3], fixedRightEyePos[3], thisLeftEyePos[3], thisRightEyePos[3], accLeftEyePos[3] = { 0.3,0.3,0.2 }, accRightEyePos[3] = { 0.7,0.3,0.2 };//position relative to face image
	double accFacePos[3];
	double avgEye = 8, avgFace = 4;//use this number to average eye position
	double eyeArea[5] = { 0,0.5,0.25,0.5,0.25 };//the area where eye is searched. 0. x of left eye, 1. x of right eye, 2. y of eyes, 3. eye with 4. eye height
	double noseArea[4] = { 0.25,0.5,0.5,0.25 };// x,y,width,height
	double mouthArea[4] = { 0.25,0.65,0.5,0.35 };// x,y,width,height
	double eyeEnlarge = 1; //the actual area for detecting eye is enlarged. value over than 1 may cause problems
	double faceRej = 0.5;//face smaller than this is rejected. cur_face/acc_face
	double faceMov = 0.05;//ratio for face moving
	double minEye = 0.15;//minimum eye size
	double fixedEyeRatio = 0.9;
	int alertThold1 = 80;
	int alertThold2 = 90;
	int alertMax = 100;
	int alertStep = 2;
	int drowsiness = 0;
	int frameCount = 0;
	int frameSkipped = 3;
	bool alertOn = true;
	bool drowsyOn = true;
	bool yawningDetect = true;
	bool bwFrame = false;
	bool noseDetect = false;
	bool faceMoving = false;
	bool noFace = true;
	bool fixEye = false;//flag for fixing eyes
	bool copyEye = false;//flag for copying eye postion to fixed eye position
	bool skipFrame = true;//do a face detection on frameSkiped

	std::string const basePath = "C:\\Users\\WU ZIRUI\\Dropbox\\FinalYear\\FYP\\";
	cv::Mat face;
	cv::CascadeClassifier faceCascade, eyeCascade;
	cv::Ptr<cv::SimpleBlobDetector> eyeBlobPtr, mouthBlobPtr, noseBlobPtr;
	cv::SimpleBlobDetector::Params eyeBlobParams, mouthBlobParams, noseBlobParams;
};

Driver::Driver() {//Constructor, initilize all relevant files and pointers
	faceCascade.load(basePath + "xml\\haarcascade_frontalface_alt.xml");
	eyeCascade.load(basePath + "xml\\haarcascade_eye.xml");
	BlobThold(1, 99, 1);
	BlobThold(1, 110, 2);
	BlobThold(1, 129, 3);
}

bool Driver::FindFace(cv::Mat frame) {
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(frame, frame);
	std::vector<cv::Rect> vecRect;
	int idx = 0;
	faceCascade.detectMultiScale(frame, vecRect, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	if (vecRect.size() == 0) return false;//no face detected or more than 1 faces!
	if (vecRect.size() > 1) if (vecRect[0].width < vecRect[1].width) idx = 1;//choose the larger one
	if (vecRect[idx].width / (accFacePos[2] * frame.cols) < faceRej) return false;//face is rejected due to sudden change!
	if ((abs(accFacePos[0] - vecRect[idx].x / (double)frame.cols) > faceMov) || (abs(accFacePos[1] - vecRect[idx].y / (double)frame.rows) > faceMov)) faceMoving = true;
	else faceMoving = false;//check if face is moving and set the flag
	accFacePos[0] = ((avgFace - 1) / avgFace)*accFacePos[0] + (1 / avgFace)*vecRect[idx].x / frame.cols;
	accFacePos[1] = ((avgFace - 1) / avgFace)*accFacePos[1] + (1 / avgFace)*vecRect[idx].y / frame.rows;
	accFacePos[2] = ((avgFace - 1) / avgFace)*accFacePos[2] + (1 / avgFace)*vecRect[idx].width / frame.cols;
	face = frame(cv::Rect(frame.cols*accFacePos[0], frame.rows*accFacePos[1], frame.cols*accFacePos[2], frame.cols*accFacePos[2]));
	return true;
}

bool Driver::FindEye(bool isLeft) {
	//if (face.empty()) return false;//Only time there is no eye is when there is no face!
	std::vector<cv::Rect> vecRect;
	cv::Rect tmpRect;
	double wid = face.rows;
	if (isLeft) tmpRect = cv::Rect(eyeArea[0] * wid, eyeArea[2] * wid, eyeArea[3] * wid, eyeArea[4] * wid);//area contains left eye
	else tmpRect = cv::Rect(eyeArea[1] * wid, eyeArea[2] * wid, eyeArea[3] * wid, eyeArea[4] * wid);//area contains right eye
	eyeCascade.detectMultiScale(face(tmpRect), vecRect, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	if (vecRect.size() == 0) {//cascade fails to find eye. eye location is guessed in this frame
		if (isLeft) for (int i = 0; i < 3; i++) thisLeftEyePos[i] = accLeftEyePos[i];
		else for (int i = 0; i < 3; i++) thisRightEyePos[i] = accRightEyePos[i];
	}
	else {//cascade finds eye
		if (isLeft) { thisLeftEyePos[0] = eyeArea[0] + vecRect[0].x / wid, thisLeftEyePos[1] = eyeArea[2] + vecRect[0].y / wid, thisLeftEyePos[2] = vecRect[0].width / wid; }
		else { thisRightEyePos[0] = eyeArea[1] + vecRect[0].x / wid, thisRightEyePos[1] = eyeArea[2] + vecRect[0].y / wid, thisRightEyePos[2] = vecRect[0].width / wid; }
	}
	return true;
}

bool Driver::BlobEye(bool isLeft) {
	cv::Mat eye;
	std::vector<cv::KeyPoint> vecKeyPoints;
	int wid = face.rows;
	if (isLeft) eye = face(cv::Rect(wid*thisLeftEyePos[0], wid*thisLeftEyePos[1], wid*thisLeftEyePos[2] * eyeEnlarge, wid*thisLeftEyePos[2] * eyeEnlarge));//may cause over flow
	else eye = face(cv::Rect(wid*thisRightEyePos[0], wid*thisRightEyePos[1], wid*thisRightEyePos[2] * eyeEnlarge, wid*thisRightEyePos[2] * eyeEnlarge));
	cv::equalizeHist(eye, eye);
	eyeBlobPtr->detect(eye, vecKeyPoints);
	if (vecKeyPoints.size() > 0) {
		if (isLeft) for (int i = 0; i < 3; i++) accLeftEyePos[i] = ((avgEye - 1) / avgEye)*accLeftEyePos[i] + (1 / avgEye)*thisLeftEyePos[i];
		if ((!isLeft)) for (int i = 0; i < 3; i++) accRightEyePos[i] = ((avgEye - 1) / avgEye)*accRightEyePos[i] + (1 / avgEye)*thisRightEyePos[i];
	}
	//cv::equalizeHist(eye, eye);
	if (copyEye) {
		for (int i = 0; i < 3; i++) fixedLeftEyePos[i] = accLeftEyePos[i];
		for (int i = 0; i < 3; i++) fixedRightEyePos[i] = accRightEyePos[i];
		copyEye = false;
	}
	if (fixEye) {
		if (isLeft) for (int i = 0; i < 3; i++) accLeftEyePos[i] = fixedEyeRatio * fixedLeftEyePos[i] + (1 - fixedEyeRatio)*accLeftEyePos[i];
		else for (int i = 0; i < 3; i++) accRightEyePos[i] = fixedEyeRatio * fixedRightEyePos[i] + (1 - fixedEyeRatio)*accRightEyePos[i];
	}

	if (accLeftEyePos[2] < minEye) accLeftEyePos[2] = minEye;
	if (accRightEyePos[2] < minEye) accRightEyePos[2] = minEye;

	if (isLeft) eye = face(cv::Rect(wid*accLeftEyePos[0], wid*accLeftEyePos[1], wid*accLeftEyePos[2] * eyeEnlarge, wid*accLeftEyePos[2] * eyeEnlarge));//may cause over flow
	else eye = face(cv::Rect(wid*accRightEyePos[0], wid*accRightEyePos[1], wid*accRightEyePos[2] * eyeEnlarge, wid*accRightEyePos[2] * eyeEnlarge));
	eyeBlobPtr->detect(eye, vecKeyPoints);
	if (vecKeyPoints.size() > 0) return true;
	else return false;
}

cv::Mat Driver::Detect(cv::Mat frame) {
	frameCount++;
	if (bwFrame) {
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
	}
	int barLen = 140, barHet = 18, offSet = 8;
	cv::rectangle(frame, cv::Point(frame.cols - offSet - barLen, offSet), cv::Point(frame.cols - offSet, offSet + barHet), cv::Scalar(0, 0, 0), 1);
	cv::rectangle(frame, cv::Rect(frame.cols - offSet - barLen + 2, offSet + 3, drowsiness*barLen / 100 - 2, barHet - 5), cv::Scalar(120, 255, 255), -1);
	cv::putText(frame, "drowsiness:", cv::Point(frame.cols - barLen - 110, offSet + 15), 1, 1, cv::Scalar(0, 0, 0));
	cv::putText(frame, std::to_string(drowsiness) + "%", cv::Point(frame.cols - barLen, offSet + 15), 1, 1, cv::Scalar(0, 0, 0));
	if (skipFrame) {
		if (frameCount % frameSkipped == 1) {//find face on every other frame
			if ((!FindFace(frame))) {
				noFace = true;
				return frame;//nothing is detected;
			}
			else noFace = false;
		}
		else if (noFace) return frame;
	}
	else {
		if ((!FindFace(frame))) return frame;//nothing is detected
	}
	if (faceMoving) {
		cv::rectangle(frame, cv::Rect(frame.cols*accFacePos[0], frame.rows*accFacePos[1], frame.cols*accFacePos[2], frame.cols*accFacePos[2]), cv::Scalar(0, 0, 0), 1);
		return frame;
	}
	FindEye(true);
	FindEye(false);
	int wid = face.rows;
	leftStats <<= 1;
	rightStats <<= 1;
	headStats <<= 1;
	mouthStats <<= 1;
	if (BlobEye(true)) leftStats |= 0x01;//left open. 1 for open 0 for closed
	else leftStats &= 0xfe;//last bit set to 0 because it is closed
	if (BlobEye(false)) rightStats |= 0x01;//right open
	else rightStats &= 0xfe;
	BlobMouth();
	cv::Rect facePos = cv::Rect(frame.cols*accFacePos[0], frame.rows*accFacePos[1], frame.cols*accFacePos[2], frame.cols*accFacePos[2]);
	if (yawningDetect) {
		if (BlobMouth()) mouthStats |= 0x01;//1 open(yawning) 0 closed
		else mouthStats &= 0xfe;
		if (mouthStats & 0xff != 0) {
			cv::putText(frame, "!yawning alert!", cv::Point(12, 23), 1, 1, cv::Scalar(0, 0, 255));
			cv::rectangle(frame, cv::Rect(facePos.x + mouthArea[0] * wid, facePos.y + mouthArea[1] * wid, mouthArea[2] * wid, mouthArea[3] * wid), cv::Scalar(0, 0, 0), 1);
		}
	}
	if (noseDetect) {
		if (BlobNose()) headStats |= 0x01;// looking straight
		else headStats &= 0xfe;// looking aside
		cv::rectangle(frame, facePos, cv::Scalar(0, 0, 0), headStats & 0x0f ? 1 : 2);
	}
	else cv::rectangle(frame, facePos, cv::Scalar(0, 0, 0), 1);
	//draw eyes
	cv::rectangle(frame, cv::Rect(facePos.x + accLeftEyePos[0] * wid, facePos.y + accLeftEyePos[1] * wid, accLeftEyePos[2] * wid, accLeftEyePos[2] * wid), cv::Scalar(0, 0, 0), leftStats & 0x0f ? 1 : 2);
	cv::rectangle(frame, cv::Rect(facePos.x + accRightEyePos[0] * wid, facePos.y + accRightEyePos[1] * wid, accRightEyePos[2] * wid, accRightEyePos[2] * wid), cv::Scalar(0, 0, 0), rightStats & 0x0f ? 1 : 2);
	if (drowsyOn) {
		if ((leftStats & 0x0f) && (rightStats & 0x0f)) drowsiness -= alertStep;//both open
		if (!(leftStats & 0x0f) && !(rightStats & 0x0f)) drowsiness += alertStep;//both closed
		drowsiness = BoundNum(drowsiness, 0, alertMax);
	}
	else drowsiness = 0;
	if (alertOn) {
		if (drowsiness > alertThold2) {
			cv::putText(frame, "WAKE", cv::Point(80, 210), 1, 10, cv::Scalar(19 * frameCount % 255, 23 * frameCount % 255, 17 * frameCount % 255), 10);
			cv::putText(frame, "UP!!!!", cv::Point(100, 310), 1, 10, cv::Scalar(23 * frameCount % 255, 17 * frameCount % 255, 19 * frameCount % 255), 10);
		}
		else if (drowsiness>alertThold1)
			cv::putText(frame, "ALERT!", cv::Point(40, 260), 1, 10, cv::Scalar(2 * frameCount % 255, 3 * frameCount % 255, 5 * frameCount % 255), 10);
	}
	return frame;
}

void Driver::BlobThold(int min, int max, int select) {
	if (select == 1) {
		if (min > 0) eyeBlobParams.minThreshold = min;//zero for not changing
		if (max > 0) eyeBlobParams.maxThreshold = max;
		eyeBlobPtr = cv::SimpleBlobDetector::create(eyeBlobParams);
	}
	if (select == 2) {
		if (min > 0) mouthBlobParams.minThreshold = min;
		if (max > 0) mouthBlobParams.maxThreshold = max;
		mouthBlobParams.filterByInertia = true;
		mouthBlobParams.minInertiaRatio = 0.01;
		//mouthBlobParams.filterByCircularity = true;
		//mouthBlobParams.minCircularity = 0.1;
		//.filterByConvexity =true;
		//mouthBlobParams.minConvexity = 0.7;
		mouthBlobPtr = cv::SimpleBlobDetector::create(mouthBlobParams);
	}
	if (select == 3) {
		if (min > 0) noseBlobParams.minThreshold = min;
		if (max > 0) noseBlobParams.maxThreshold = max;
		noseBlobPtr = cv::SimpleBlobDetector::create(noseBlobParams);
	}

	return;
}

bool Driver::BlobNose() {//true for head ahead false for looking aside
	int wid = face.rows;
	cv::Mat nose = face(cv::Rect(noseArea[0] * wid, noseArea[1] * wid, noseArea[2] * wid, noseArea[3] * wid));
	std::vector<cv::KeyPoint> vecKeyPoints;
	//cv::equalizeHist(nose, nose);
	eyeBlobPtr->detect(nose, vecKeyPoints);
	//cv::drawKeypoints(nose, vecKeyPoints, nose, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::imshow("Nose", nose);
	if (vecKeyPoints.size() > 1) return true;
	else return false;
}

void Driver::setParams(int thold1, int thold2, int step, int drowsy, double eye) {//thold1 thold2 step drowsiness minEye
	if (thold1 > -1) alertThold1 = BoundNum(thold1, 1, 99);
	if (thold2 > -1) alertThold2 = BoundNum(thold2, 1, 99);
	if (step > -1) alertStep = BoundNum(step, 1, 11);
	if (drowsy > -1) drowsiness = BoundNum(step, 0, 100);
	if (eye > -1) minEye = BoundNum(eye, 0.1, 0.4);
}

void Driver::flipParams(bool alert, bool drowsy, bool bw, bool nose, bool mouth, bool eyefix,bool skip) {//alertOn drowsyOn bwFrame noseDetect
	if (alert) alertOn = !alertOn;
	if (drowsy) drowsyOn = !drowsyOn;
	if (bw) bwFrame = !bwFrame;
	if (nose) noseDetect = !noseDetect;
	if (mouth) yawningDetect = !yawningDetect;
	if (eyefix) {
		fixEye = !fixEye;
		if (fixEye) copyEye = true;
	}
	if (skip) skipFrame = !skipFrame;
}
double Driver::BoundNum(double nm, double mn, double mx) {
	if (nm > mx) return mx;//bigger than maxium number
	if (nm < mn) return mn;//return minunal number
	return nm;
}

bool Driver::BlobMouth() {
	int wid = face.rows;
	cv::Mat mouth = face(cv::Rect(mouthArea[0] * wid, mouthArea[1] * wid, mouthArea[2] * wid, mouthArea[3] * wid));
	cv::equalizeHist(mouth, mouth);
	std::vector<cv::KeyPoint> vecKeyPoints;
	mouthBlobPtr->detect(mouth, vecKeyPoints);
	for (int i = 0; i < vecKeyPoints.size(); i++)
		if ((double)vecKeyPoints[i].size / face.rows > 0.06) return true; //open mouth
	return false; //closed mouth
}
int main() {
	cv::VideoCapture cap;
	cap.open(0);
	cv::Mat frame;
	Driver Linus;
	while (1) {
		cap >> frame;
		cv::imshow("Driver's cam", Linus.Detect(frame));
		char tmp = (char)cv::waitKey(10);
		if (tmp == 'r') Linus.setParams(-1, -1, -1, 0, -1);
		if (tmp == 'f') Linus.flipParams(0, 0, 0, 0, 0, 0,1);
	}
	return 0;
}