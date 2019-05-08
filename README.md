# simple_CNN

this netural network program based c++ and opencv;

API Example:
Load model and perform prediction:
```
#include "simple_cnn.h"
cnn.read("../models/mnist_quickstart.txt");

```

API Example: Construction of a new CNN for MNIST, and train records:  
```
#include "simple_cnn.h"

Simple_cnn::network cnn(solver.c_str());
cnn.set_mini_batch_size(mini_batch_size);
cnn.set_smart_training(true); // automate training
cnn.set_learning_rate(initial_learning_rate);
	
cnn.push_back("I1", "input 32 32 3");				// CIFAR is 32x32x3
cnn.push_back("C1", "convolution 3 16 1 elu");		// 32-3+1=30
cnn.push_back("P1", "max_pool 3 3");	// 10x10 out
cnn.push_back("C2", "convolution 3 64 1 elu");		// 8x8 out
cnn.push_back("P2", "max_pool 4 4");	// 2x2 out
cnn.push_back("FC2", "softmax 10");
 
cnn.connect_all(); // connect layers automatically (no branches)

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

```
