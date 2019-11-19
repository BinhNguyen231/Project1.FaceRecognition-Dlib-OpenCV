
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <iostream>
#include <time.h>


using namespace std;
using namespace dlib;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;


int main(int argc, char **argv)
{
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor sp;
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	typedef matrix<float, 0, 1> sample_type;
	typedef radial_basis_kernel<sample_type> kernel_type;

	typedef decision_function<kernel_type> dec_funct_type;
	typedef normalized_function<dec_funct_type> funct_type;

	funct_type learned_function;

	//file .dat da duoc huan luyen tu truoc
	deserialize("saved_function1.dat") >> learned_function;
	// print out the number of support vectors in the resulting decision function
	cout << "\nnumber of support vectors in our learned_function is "
		<< learned_function.function.basis_vectors.size() << endl;

	////////////////////////////////////////////

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}
	image_window win;
	int totalFrame = 0;

	std::vector<dlib::correlation_tracker> trackers;
	std::vector<double> labels;
	std::vector<dlib::rectangle> rects;
	while (!win.is_closed())
	{
		clock_t start = clock();
		std::vector<matrix<rgb_pixel>> faces;

		// Grab a frame
		cv::Mat temp;
		if (!cap.read(temp))
		{
			break;
		}
		float scale = 300.0 / temp.rows;
		cv::Mat resized;
		cv::resize(temp, resized, cv::Size(int(temp.cols*scale), int(temp.rows*scale)));
		cv_image<bgr_pixel> img(resized);


		// Detect faces 
		if (totalFrame % 5 == 0) {
			trackers.clear();
			labels.clear();
			rects.clear();
			for (auto face : detector(img))
			{
				auto shape = sp(img, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));

				rects.push_back(face);
			}
			std::vector<sample_type> face_descriptors = net(faces);
			for (int i = 0; i < face_descriptors.size(); ++i)
			{
				//thuc hien gan nhan cho cac khuon mat nhan dien duoc
				//neu la toi se duoc gan la "Binh"
				//neu la nguoi khac se duoc gan la "Other"
				cv::Point left_top;
				cv::Point right_bot;
				left_top.x = int(rects[i].left()) / scale;
				left_top.y = int(rects[i].top()) / scale;
				right_bot.x = int(rects[i].right()) / scale;
				right_bot.y = int(rects[i].bottom()) / scale;
				cv::rectangle(temp, left_top, right_bot, cv::Scalar(0, 255, 0), 1, 8, 0);
				if (learned_function(face_descriptors[i]) > 0)
				{
					cv::putText(temp, "Binh", left_top, cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv::LINE_AA);
					labels.push_back(1);
				}
				else {
					cv::putText(temp, "Other", left_top, cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv::LINE_AA);
					labels.push_back(0);
				}
				dlib::correlation_tracker t;
				t.start_track(img, rects[i]);
				trackers.push_back(t);
			}
		}
		else {
			for (int j = 0; j < trackers.size(); j++) {
				trackers[j].update(img);
				dlib::rectangle pos = trackers[j].get_position();
				int startX = int(pos.left() / scale);
				int	startY = int(pos.top() / scale);
				int	endX = int(pos.right() / scale);
				int	endY = int(pos.bottom() / scale);
				cv::rectangle(temp, cv::Point(startX, startY), cv::Point(endX, endY), cv::Scalar(0, 255, 0), 1, 8, 0);
				if (labels[j] == 1)
					cv::putText(temp, "Binh", cv::Point(startX, startY), cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv::LINE_AA);
				else
					cv::putText(temp, "Other", cv::Point(startX, startY), cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv::LINE_AA);
			}
		}
		cv_image<bgr_pixel> img2(temp);
		totalFrame++;
		clock_t end = clock();
		float fps = 1000.0 / (end - start);
		char cfps[20];
		memset(cfps, 0, 20);
		sprintf(cfps, "FPS: %.6lf", fps);
		cv::putText(temp, cfps, cv::Point(40, 40), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv::LINE_AA);
		win.clear_overlay();
		win.set_title("Test");
		win.set_image(img2);

		totalFrame++;
	}

	system("pause");
	return 0;
}
