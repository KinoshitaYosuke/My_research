// hog.cpp
// (C) Toshihiko YAMASAKI
// 本プログラムを使用したことによるいかなる障害、損害も責任はとりません
// 各自の責任において使用してください。
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "svm.h"

#define YUDO_CD 0.7
#define YUDO_FD 0.999

using namespace std;

struct svm_node *x;
int max_nr_attr = 64;

double FD_max_ans = 0.0;
int FD_max_XY = 0;
//int CD_max_X = 0;
//int CD_max_Y = 0;

struct svm_model* CD;
struct svm_model* FD_20x36;
struct svm_model* FD_20x40;
struct svm_model* FD_24x40;
struct svm_model* FD_24x44;
struct svm_model* FD_24x48;
struct svm_model* FD_28x48;
struct svm_model* FD_28x52;
struct svm_model* FD_28x56;
struct svm_model* FD_32x52;
struct svm_model* FD_32x56;
struct svm_model* FD_32x60;
struct svm_model* FD_32x64;
struct svm_model* FD_36x60;
struct svm_model* FD_36x64;
struct svm_model* FD_40x64;
struct svm_model* FD_16x44;
struct svm_model* FD_16x48;
struct svm_model* FD_16x52;
struct svm_model* FD_16x56;
struct svm_model* FD_16x60;
struct svm_model* FD_16x64;
struct svm_model* FD_20x56;
struct svm_model* FD_20x60;
struct svm_model* FD_20x64;
struct svm_model* FD_24x64;

//static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

double predict(float *hog_vector, int hog_dim, svm_model* Detector)
{
	int svm_type = svm_get_svm_type(Detector);
	int nr_class = svm_get_nr_class(Detector);
	double *prob_estimates = NULL;
	int j;

	int *labels = (int *)malloc(nr_class*sizeof(int));
	svm_get_labels(Detector, labels);
	prob_estimates = (double *)malloc(nr_class*sizeof(double));
	free(labels);


	max_line_len = 1024;
	//	line = (char *)malloc(max_line_len*sizeof(char));
	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	int i;
	double target_label, predict_label;
	char *idx, *val, *label, *endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0


	//	for (i = 0; hog_vector[i] != NULL; i++)
	for (i = 0; i < hog_dim; i++)
	{
		//		clog << i << endl;
		if (i >= max_nr_attr - 1)	// need one more for index = -1
		{
			max_nr_attr *= 2;
			x = (struct svm_node *) realloc(x, max_nr_attr*sizeof(struct svm_node));
		}

		//			clog << i << endl;

		errno = 0;
		x[i].index = i;
		inst_max_index = x[i].index;

		errno = 0;
		x[i].value = hog_vector[i];
		//		cout << i << ":" << hog_vector[i] << endl;
	}
	x[i].index = -1;


	predict_label = svm_predict_probability(Detector, x, prob_estimates);
	//	if (prob_estimates[0] >= YUDO_CD)
	//		printf(" %f\n", prob_estimates[0]);

	free(x);
	//	free(line);

	return prob_estimates[0];
}

int minimum(int a, int b){
	if (a<b){
		return a;
	}
	return b;
}

void get_HOG(cv::Mat im, float* hog_vector){
	int cell_size = 6;
	int rot_res = 9;
	int block_size = 3;
	int x, y, i, j, k, m, n, count;
	float dx, dy;
	float ***hist, *vec_tmp;
	float norm;
	CvMat *mag = NULL, *theta = NULL;
	//	FILE *hog_hist;

	//	fopen_s(&hog_hist,"d_im_hog.txt", "w");
	//	fopen_s(&hog_hist, "d_im_hog.bin", "w");
	//	fprintf(hog_hist, "%c ", '1');
	int counter = 1;

	mag = cvCreateMat(im.rows, im.cols, CV_32F);
	theta = cvCreateMat(im.rows, im.cols, CV_32F);
	for (y = 0; y<im.rows; y++){
		for (x = 0; x<im.cols; x++){
			if (x == 0 || x == im.cols - 1 || y == 0 || y == im.rows - 1){
				cvmSet(mag, y, x, 0.0);
				cvmSet(theta, y, x, 0.0);
			}
			else{
				dx = double((uchar)im.data[y*im.step + x + 1]) - double((uchar)im.data[y*im.step + x - 1]);
				dy = double((uchar)im.data[(y + 1)*im.step + x]) - double((uchar)im.data[(y - 1)*im.step + x]);
				cvmSet(mag, y, x, sqrt(dx*dx + dy*dy));
				cvmSet(theta, y, x, atan(dy / (dx + 0.01)));
			}
		}
	}

	// histogram generation for each cell
	hist = (float***)malloc(sizeof(float**) * (int)ceil((float)im.rows / (float)cell_size));
	if (hist == NULL) exit(1);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++){
		hist[i] = (float**)malloc(sizeof(float*)*(int)ceil((float)im.cols / (float)cell_size));
		if (hist[i] == NULL) exit(1);
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++){
			hist[i][j] = (float *)malloc(sizeof(float)*rot_res);
			if (hist[i][j] == NULL) exit(1);
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++){
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++){
			for (k = 0; k<rot_res; k++){
				hist[i][j][k] = 0.0;
			}
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++){
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++){
			for (m = i*cell_size; m<minimum((i + 1)*cell_size, im.rows); m++){
				for (n = j*cell_size; n<minimum((j + 1)*cell_size, im.cols); n++){
					hist[i][j][(int)floor((cvmGet(theta, m, n) + CV_PI / 2)*rot_res / CV_PI)] += cvmGet(mag, m, n);
				}
			}
		}
	}

	// normalization for each block & generate vector
	vec_tmp = (float *)malloc(sizeof(float)*block_size*block_size*rot_res);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size) - (block_size - 1); i++){
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size) - (block_size - 1); j++){
			count = 0;
			norm = 0.0;
			for (m = i; m<i + block_size; m++){
				for (n = j; n<j + block_size; n++){
					for (k = 0; k<rot_res; k++){
						vec_tmp[count++] = hist[m][n][k];
						norm += hist[m][n][k] * hist[m][n][k];
					}
				}
			}
			for (count = 0; count<block_size*block_size*rot_res; count++){
				vec_tmp[count] = vec_tmp[count] / (sqrt(norm + 1));
				if (vec_tmp[count]>0.2) vec_tmp[count] = 0.2;
				//		fprintf(hog_hist, "%d:%.4f ", counter, vec_tmp[count]);
				hog_vector[counter] = vec_tmp[count];
				//		cout << counter << ":" << hog_vector[counter] << endl;
				//		printf("%d:%.4f ",counter, vec_tmp[count]);
				counter++;
			}
		}
	}
	//	printf("\n");
	//	fprintf(hog_hist, "\n");
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j <(int)ceil((float)im.cols / (float)cell_size); j++) {
			free(hist[i][j]);
		}
		free(hist[i]);
	}
	free(hist);
	cvReleaseMat(&mag);
	cvReleaseMat(&theta);
	//	fclose(hog_hist);
}

cv::Mat draw_rectangle(cv::Mat ans_im, int x, int y, int width, int height, int r, int g, int b){
	rectangle(ans_im, cvPoint(x, y), cvPoint(x + width, y + height), CV_RGB(r, g, b), 1);
	return ans_im;
}

int dimension(int x, int y){

	return (int)(81 * ((int)ceil((float)x / 6) - 2) * ((int)ceil((float)y / 6) - 2));

	if (x % 6 == 0){
		return 81 * (x / 6 - 2) * (y / 6 - 1);
	}
	if (y % 6 == 0){
		return 81 * (x / 6 - 1) * (y / 6 - 2);
	}
	if (x % 6 == 0 && y % 6 == 0){
		return 81 * (x / 6 - 2) * (y / 6 - 2);
	}
	else
		return 81 * (x / 6 - 1) * (y / 6 - 1);
}

void FD_predict(int width, int height, cv::Mat FD_img, svm_model* Detector){
	cv::Mat FD_cut_im(FD_img, cv::Rect(32 - width / 2, 32 - height / 2, width, height));
	//	cv::Mat FD_max_img = FD_cut_im;
	//	cvtColor(FD_cut_im, FD_cut_im, CV_RGB2GRAY);
	int hog_dim = dimension(FD_cut_im.cols, FD_cut_im.rows);
	float FD_vector[6562];	//各次元のHOGを格納
	get_HOG(FD_cut_im, FD_vector);	//HOGの取得

	double ans = predict(FD_vector, hog_dim, Detector);

	//	cout << "x,y:" << width << "," << height << "=" << ans << endl;

	if (FD_max_ans < ans && YUDO_FD < ans){
		FD_max_ans = ans;
		FD_max_XY = width * 100 + height;
		//	cv::imwrite("result_FD.bmp", FD_cut_im);
		cv::imshow("FD_result", FD_cut_im);
		cvWaitKey(10);
	}
	//	return FD_max_ans;
}

int main(int argc, char** argv){
	//変数宣言
	int x, y, d, count = 0;
	IplImage *im = NULL;
	time_t timer1, timer2;
	//	float yudo_max=0.0;
	//	int yudo_x=0,yudo_y=0;
	float yudo_max[100];
	int yudo_x[100], yudo_y[100];
	int max_CD_num = 0;
	cv::Mat CD_img[100];
	int hog_dim;

	//画像の取り込み
	cv::Mat ans_img_CF = cv::imread("test/test_01.bmp", 1);	//検出する画像
	cv::Mat ans_img_CD = ans_img_CF.clone();
	cv::Mat ans_img_FD = ans_img_CF.clone();
	cv::Mat img;			//検出矩形処理を施す画像
	cvtColor(ans_img_CF, img, CV_RGB2GRAY);
	cv::imwrite("test_gray.bmp", img);

	//CoarseDetectorの取り込み
	if ((CD = svm_load_model("model_file/CD_3.model")) == 0)exit(1);
	if ((FD_20x36 = svm_load_model("model_file/FD_squat_20x36.model")) == 0)exit(1);
	if ((FD_20x40 = svm_load_model("model_file/FD_squat_20x40.model")) == 0)exit(1);
	if ((FD_24x40 = svm_load_model("model_file/FD_squat_24x40.model")) == 0)exit(1);
	if ((FD_24x44 = svm_load_model("model_file/FD_squat_24x44.model")) == 0)exit(1);
	if ((FD_24x48 = svm_load_model("model_file/FD_squat_24x48.model")) == 0)exit(1);
	if ((FD_28x48 = svm_load_model("model_file/FD_squat_28x48.model")) == 0)exit(1);
	if ((FD_28x52 = svm_load_model("model_file/FD_squat_28x52.model")) == 0)exit(1);
	if ((FD_28x56 = svm_load_model("model_file/FD_squat_28x56.model")) == 0)exit(1);
	if ((FD_32x52 = svm_load_model("model_file/FD_squat_32x52.model")) == 0)exit(1);
	if ((FD_32x56 = svm_load_model("model_file/FD_squat_32x56.model")) == 0)exit(1);
	if ((FD_32x60 = svm_load_model("model_file/FD_squat_32x60.model")) == 0)exit(1);
	if ((FD_32x64 = svm_load_model("model_file/FD_squat_32x64.model")) == 0)exit(1);
	if ((FD_36x60 = svm_load_model("model_file/FD_squat_36x60.model")) == 0)exit(1);
	if ((FD_36x64 = svm_load_model("model_file/FD_squat_36x64.model")) == 0)exit(1);
	if ((FD_40x64 = svm_load_model("model_file/FD_squat_40x64.model")) == 0)exit(1);
	if ((FD_16x44 = svm_load_model("model_file/FD_stand_16x44.model")) == 0)exit(1);
	if ((FD_16x48 = svm_load_model("model_file/FD_stand_16x48.model")) == 0)exit(1);
	if ((FD_16x52 = svm_load_model("model_file/FD_stand_16x52.model")) == 0)exit(1);
	if ((FD_16x56 = svm_load_model("model_file/FD_stand_16x56.model")) == 0)exit(1);
	if ((FD_16x60 = svm_load_model("model_file/FD_stand_16x60.model")) == 0)exit(1);
	if ((FD_16x64 = svm_load_model("model_file/FD_stand_16x64.model")) == 0)exit(1);
	if ((FD_20x56 = svm_load_model("model_file/FD_stand_20x56.model")) == 0)exit(1);
	if ((FD_20x60 = svm_load_model("model_file/FD_stand_20x60.model")) == 0)exit(1);
	if ((FD_20x64 = svm_load_model("model_file/FD_stand_20x64.model")) == 0)exit(1);
	if ((FD_24x64 = svm_load_model("model_file/FD_stand_24x64.model")) == 0)exit(1);

	clog << "model load complete" << endl;

	timer1 = clock();

	//Coarse Detectorによる人物検出
	for (y = 0; (y + 64) <= ans_img_CF.rows; y += 4){
		for (x = 0; (x + 64) <= ans_img_CF.cols; x += 4){
			//	for (y = 2; (y + 66) <= ans_img_CF.rows; y += 4){
			//		for (x = 2; (x + 66) <= ans_img_CF.cols; x += 4){
			cv::Mat d_im(img, cv::Rect(x, y, 64, 64));		//検出領域のみにする
			hog_dim = dimension(d_im.cols, d_im.rows);
			float hog_vector[6562];							//各次元のHOGを格納
			get_HOG(d_im, hog_vector);	//HOGの取得
			double ans = predict(hog_vector, hog_dim, CD);	//尤度の算出
			if (ans >= YUDO_CD){//尤度から人物か非人物かの判断
				ans_img_CF = draw_rectangle(ans_img_CF, x, y, 64, 64, 255, 0, 0);
				ans_img_CD = draw_rectangle(ans_img_CD, x, y, 64, 64, 255, 0, 0);
				/*		//最大値をとる領域のみを採用する場合----------------------------------------------------------------------------
				if (yudo_max <= ans){
				yudo_max = ans;
				yudo_x = x;
				yudo_y = y;
				cv::Mat max_img(ans_img,cv::Rect(x,y,64,64));
				cv::imwrite("max_img.bmp", max_img);
				}
				//-------------------------------------------------------------------------------------------------------------
				*/
				//一定以上の尤度を持つ場合全て---------------------------------------------------------------------------------
				yudo_max[count] = ans;
				yudo_x[count] = x;
				yudo_y[count] = y;
				CD_img[count] = d_im.clone();
				//周辺ピクセル含めてトリミング
				//		CD_img[count] = img.clone();
				//		CD_img[count] = CD_img[count](cv::Rect(x-2, y-2, 68, 68));
				//				cout << count << ":" << ans << endl;
				//				cout << x << "," << y << endl;
				cv::imwrite("max_img.bmp", CD_img[count]);
				//		cv::imshow("CD result", CD_img[count]);
				//		cvWaitKey(0);

				//-------------------------------------------------------------------------------------------------------------

				count++;
			}
		}
	}
	//	ans_img = draw_rectangle(ans_img, yudo_x, yudo_y, 64, 64, 255, 0, 0);


	timer2 = clock();
	clog << timer2 - timer1 << "[mmsec]" << endl;

	//	cv::imwrite("result_CD.bmp", ans_img);


	for (int i = 0; i < count; i++){
		//	cv::Mat FD_img = cv::imread("max_img.bmp",1);
		//	cv::Mat FD_max = FD_img;
		float zure_yudo[25];
		int zure_count = 0;
		FD_max_ans = 0.0;
		FD_max_XY = 0;
		//		for (int a = -2; a <= 2; a++){
		//			for (int b = -2; b <= 2; b++){

		//				cv::Mat FD_img = CD_img[i](cv::Rect(a + 2, b + 2, 64, 64));

		cv::Mat FD_img = CD_img[i];

		FD_predict(20, 36, FD_img, FD_20x36);
		FD_predict(20, 40, FD_img, FD_20x40);
		FD_predict(24, 40, FD_img, FD_24x40);
		FD_predict(24, 44, FD_img, FD_24x44);
		FD_predict(24, 48, FD_img, FD_24x48);
		FD_predict(28, 48, FD_img, FD_28x48);
		FD_predict(28, 52, FD_img, FD_28x52);
		FD_predict(28, 56, FD_img, FD_28x56);
		FD_predict(32, 52, FD_img, FD_32x52);
		FD_predict(32, 56, FD_img, FD_32x56);
		FD_predict(32, 60, FD_img, FD_32x60);
		FD_predict(32, 64, FD_img, FD_32x64);
		FD_predict(36, 60, FD_img, FD_36x60);
		FD_predict(36, 64, FD_img, FD_36x64);
		FD_predict(40, 64, FD_img, FD_40x64);
		FD_predict(16, 44, FD_img, FD_16x44);
		FD_predict(16, 48, FD_img, FD_16x48);
		FD_predict(16, 52, FD_img, FD_16x52);
		FD_predict(16, 56, FD_img, FD_16x56);
		FD_predict(16, 60, FD_img, FD_16x60);
		FD_predict(16, 64, FD_img, FD_16x64);
		FD_predict(20, 56, FD_img, FD_20x56);
		FD_predict(20, 60, FD_img, FD_20x60);
		FD_predict(20, 64, FD_img, FD_20x64);
		FD_predict(24, 64, FD_img, FD_24x64);

		cout << FD_max_ans << endl;
		//				zure_yudo[zure_count] = FD_max_ans;
		//				zure_count++;
		if (FD_max_ans != 0){
			ans_img_CF = draw_rectangle(ans_img_CF, yudo_x[i] + 32 - (FD_max_XY / 100) / 2, yudo_y[i] + 32 - (FD_max_XY % 100) / 2, FD_max_XY / 100, FD_max_XY % 100, 0, 255, 0);
			ans_img_FD = draw_rectangle(ans_img_FD, yudo_x[i] + 32 - (FD_max_XY / 100) / 2, yudo_y[i] + 32 - (FD_max_XY % 100) / 2, FD_max_XY / 100, FD_max_XY % 100, 0, 255, 0);
			//	ans_img = draw_rectangle(ans_img, a + yudo_x[i] + 32 - (FD_max_XY / 100) / 2, b + yudo_y[i] + 32 - (FD_max_XY % 100) / 2, FD_max_XY / 100, FD_max_XY % 100, 0, 0, 255);
			//	cv::imwrite("result_CD+FD.bmp", ans_img);
		}
		//			}
		//		}

		//		float max;
		//		int ab=0;
		//		max = zure_yudo[0];
		//		for (int z = 0; z < 25; z++){
		//			if (zure_yudo[z] > max){
		//		//		cout << zure_yudo[z] << endl;
		//				max = zure_yudo[z];
		//				ab = z;
		//			}
		//		}

		//		if (max > YUDO_FD){
		//			cout << (int)(ab / 5) - 2 + yudo_x[i] + 32 - (FD_max_XY / 100) / 2 << endl;
		//			cout << (int)(ab % 5) - 2 + yudo_y[i] + 32 - (FD_max_XY % 100) / 2 << endl;
		//			cout << FD_max_XY << endl;
		//			cout << FD_max_XY << endl;
		//			ans_img_CF = draw_rectangle(ans_img_CF, (int)(ab / 5) - 2 + yudo_x[i] + 32 - (FD_max_XY / 100) / 2, (int)(ab % 5) - 2 + yudo_y[i] + 32 - (FD_max_XY % 100) / 2, FD_max_XY / 100, FD_max_XY % 100, 0, 0, 255);
		//			ans_img_FD = draw_rectangle(ans_img_FD, (int)(ab / 5) - 2 + yudo_x[i] + 32 - (FD_max_XY / 100) / 2, (int)(ab % 5) - 2 + yudo_y[i] + 32 - (FD_max_XY % 100) / 2, FD_max_XY / 100, FD_max_XY % 100, 0, 0, 255);

		//		}
	}
	cv::imwrite("result_CD+FD.bmp", ans_img_CF);
	cv::imwrite("result_FD.bmp", ans_img_FD);
	cv::imwrite("result_CD.bmp", ans_img_CD);

	return 0;
}
