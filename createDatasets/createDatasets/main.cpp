#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/svm.h>
#include <iostream>


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

std::vector<matrix<rgb_pixel>> processingImage(const char* fileName, frontal_face_detector detector, shape_predictor sp);


int main(int argc, char **argv)
{
	typedef matrix<float, 0, 1> sample_type;
	typedef radial_basis_kernel<sample_type> kernel_type;
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor sp;
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	//training Image in folder
	const char* fileImageOfMe = "Binh\\*.jpg";
	std::vector<matrix<rgb_pixel>> facesTrainedOfMe = processingImage(fileImageOfMe, detector, sp);

	const char* fileImageOfOthers = "Unknown\\*.pgm";
	std::vector<matrix<rgb_pixel>> facesTrainedOfOthers = processingImage(fileImageOfOthers, detector, sp);


	std::vector<sample_type> faceDescriptors = net(facesTrainedOfMe);
	std::vector<float> labels;
	for (size_t i = 0; i < faceDescriptors.size(); ++i)
	{
		labels.push_back(+1);
	}
	cout << "image of me: " << faceDescriptors.size() << endl;
	std::vector<sample_type> faceDescriptorsOfOthers = net(facesTrainedOfOthers);
	for (size_t i = 0; i < faceDescriptorsOfOthers.size(); ++i)
	{
		faceDescriptors.push_back(faceDescriptorsOfOthers[i]);
		labels.push_back(-1);
	}
	cout << "image of others: " << facesTrainedOfOthers.size() << endl;
	vector_normalizer<sample_type> normalizer;
	normalizer.train(faceDescriptors);
	// now normalize each sample
	for (unsigned long i = 0; i < faceDescriptors.size(); ++i)
	{
		faceDescriptors[i] = normalizer(faceDescriptors[i]);
	}


	randomize_samples(faceDescriptors, labels);
	const double max_nu = maximum_nu(labels);
	//doing cross validation

	svm_nu_trainer<kernel_type> trainer;
	cout << "doing cross validation" << endl;
	for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
	{
		for (double nu = 0.00001; nu < max_nu; nu *= 5)
		{
			trainer.set_kernel(kernel_type(gamma));
			trainer.set_nu(nu);
			cout << "gamma: " << gamma << "    nu: " << nu;
			cout << "   cross validation accuracy: " << cross_validate_trainer(trainer, faceDescriptors, labels, 4);
		}
	}

	trainer.set_kernel(kernel_type(0.00025));
	trainer.set_nu(0.03125);
	typedef decision_function<kernel_type> dec_funct_type;
	typedef normalized_function<dec_funct_type> funct_type;
	funct_type learned_function;
	learned_function.normalizer = normalizer;
	learned_function.function = trainer.train(faceDescriptors, labels);
	cout << "\nnumber of support vectors in our learned_function is "
		<< learned_function.function.basis_vectors.size() << endl;

	serialize("saved_function1.dat") << learned_function;

	system("pause");
	return 0;
}

std::vector<matrix<rgb_pixel>> processingImage(const char* fileName, frontal_face_detector detector, shape_predictor sp)
{
	std::vector<cv::String> fn;
	cv::glob(fileName, fn, false);

	std::vector<cv::Mat> images;
	size_t count = fn.size(); //number of jpg files in images folder

	std::vector<matrix<rgb_pixel>> faces;
	for (size_t i = 0; i < count; i++)
	{
		images.push_back(cv::imread(fn[i]));
		dlib::cv_image<dlib::bgr_pixel> img(images[i]);
		for (auto face : detector(img))
		{
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
		}
	}
	return faces;
}