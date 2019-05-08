
#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
//#include <tchar.h>


#include "simple_cnn.h"
#include "util.h"



#if DEBUG
#pragma comment(lib, "opencv_world400d")
#else
#pragma comment(lib, "opencv_world400")
#endif

// 
// #include "mnist_parser.h"
// using namespace mnist;

#include "cifar_parser.h"
using namespace cifar;

const int mini_batch_size = 24;
const float initial_learning_rate = 0.04f;
std::string solver = "adam";
/*std::string data_path="../data/mnist/";*/
std::string data_path = "../data/cifar-10-batches-bin/";


// performs validation testing
float test_train(Simple_cnn::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	// use progress object for simple timing and status updating
	Simple_cnn::progress progress((int)test_images.size(), "  testing:\t\t");

	int out_size = cnn.out_size(); // we know this to be 10 for MNIST
	int correct_predictions = 0;
	const int record_cnt = (int)test_images.size();


	for (int k = 0; k < record_cnt; k++)
	{
		const int prediction = cnn.predict_class(test_images[k].data());
		if (prediction == test_labels[k]) correct_predictions += 1;
		if (k % 1000 == 0) progress.draw_progress(k);
	}

	float accuracy = (float)correct_predictions / record_cnt * 100.f;
	return accuracy;
}


int main()
{
	// ==== parse data
	// array to hold image data (note that Simple_cnn does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	std::vector<int> test_labels;
	std::vector<std::vector<float>> train_images;
	std::vector<int> train_labels;

	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if (!parse_test_data(data_path, test_images, test_labels)) { std::cerr << "error: could not parse data.\n"; return 1; }
	if (!parse_train_data(data_path, train_images, train_labels)) { std::cerr << "error: could not parse data.\n"; return 1; }

	// ==== setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad", "adam")
	Simple_cnn::network cnn(solver.c_str());
	// !! the threading must be enabled with thread count prior to loading or creating a model !!

	cnn.set_mini_batch_size(mini_batch_size);
	cnn.set_smart_training(true); // automate training
	cnn.set_learning_rate(initial_learning_rate);

	// Note, network descriptions can be read from a text file with similar format to the API
	//cnn.read("../models/mnist_quickstart.txt");

	cnn.push_back("I1", "input 32 32 3");				// CIFAR is 32x32x3
	cnn.push_back("C1", "convolution 3 16 1 elu");		// 32-3+1=30
	cnn.push_back("P1", "max_pool 3 3");	// 10x10 out
	cnn.push_back("C2", "convolution 3 64 1 elu");		// 8x8 out
	cnn.push_back("P2", "max_pool 4 4");	// 2x2 out
	cnn.push_back("FC2", "softmax 10");


	cnn.connect_all();
	// 	

	std::cout << "==  Network Configuration  ====================================================" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

	// add headers for table of values we want to log out
	Simple_cnn::html_log log;
	log.set_table_header("epoch\ttest accuracy(%)\testimated accuracy(%)\tepoch time(s)\ttotal time(s)\tlearn rate\tmodel");
	log.set_note(cnn.get_configuration());


	// setup timer/progress for overall training
	Simple_cnn::progress overall_progress(-1, "  overall:\t\t");
	const int train_samples = (int)train_images.size();
	float old_accuracy = 0;
	while (1)
	{
		overall_progress.draw_header(data_name() + "  Epoch  " + std::to_string((long long)cnn.get_epoch() + 1), true);
		// setup timer / progress for this one epoch
		Simple_cnn::progress progress(train_samples, "  training:\t\t");
		// set loss function
		cnn.start_epoch("cross_entropy");

		// manually loop through data. batches are handled internally. if data is to be shuffled, the must be performed externally
		 // schedule dynamic to help make progress bar work correctly
		for (int k = 0; k < train_samples; k++)
		{
			cnn.train_class(train_images[k].data(), train_labels[k]);
			if (k % 1000 == 0) progress.draw_progress(k);
		}

		// draw weights of main convolution layers
#ifdef CNN_CV3
		Simple_cnn::show(Simple_cnn::draw_cnn_weights(cnn, "C1", Simple_cnn::tensorglow), 4, "C1 Weights");
		Simple_cnn::show(Simple_cnn::draw_cnn_weights(cnn, "C2", Simple_cnn::tensorglow), 4, "C2 Weights");
#endif

		cnn.end_epoch();
		float dt = progress.elapsed_seconds();
		std::cout << "  mini batch:\t\t" << mini_batch_size << "                               " << std::endl;
		std::cout << "  training time:\t" << dt << " seconds " << std::endl;
		std::cout << "  model updates:\t" << cnn.train_updates << " (" << (int)(100.f*(1. - (float)cnn.train_skipped / cnn.train_samples)) << "% of records)" << std::endl;
		std::cout << "  estimated accuracy:\t" << cnn.estimated_accuracy << "%" << std::endl;


		// ==== run testing set
		progress.reset((int)test_images.size(), "  testing out-of-sample:\t");
		float accuracy = test_train(cnn, test_images, test_labels);
		std::cout << "  test accuracy:\t" << accuracy << "% (" << 100.f - accuracy << "% error)      " << std::endl;

		// if accuracy is improving, reset the training logic that may be thinking about quitting
		if (accuracy > old_accuracy)
		{
			cnn.reset_smart_training();
			old_accuracy = accuracy;
		}

		// save model
		std::string model_file = "../models/snapshots/tmp_" + std::to_string((long long)cnn.get_epoch()) + ".txt";
		cnn.write(model_file, true);
		std::cout << "  saved model:\t\t" << model_file << std::endl << std::endl;

		// write log file
		std::string log_out;
		log_out += float2str(dt) + "\t";
		log_out += float2str(overall_progress.elapsed_seconds()) + "\t";
		log_out += float2str(cnn.get_learning_rate()) + "\t";
		log_out += model_file;
		log.add_table_row(cnn.estimated_accuracy, accuracy, log_out);
		// will write this every epoch
		log.write("../models/snapshots/Simple_cnn_mnist_log.htm");

		// can't seem to improve
		if (cnn.evalu_left_the_building())
		{
			std::cout << "Elvis just left the building. No further improvement in training found.\nStopping.." << std::endl;
			break;
		}

	};
	std::cout << std::endl;
	return 0;
}
