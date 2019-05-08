#pragma once


#include <iostream> // cout
#include <sstream>
#include <fstream>
#include <iomanip> //setw
#include <random>
#include <stdio.h>


namespace mnist
{
std::string data_name()
{
	return std::string("MNIST");
}

// from tiny_cnn
template<typename T>
T* reverse_endian(T* p) {
	std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
	return p;
}

// from tiny_cnn (kinda)
bool parse_mnist_labels(const std::string& label_file, std::vector<int> *labels) {
	std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

	if (ifs.bad() || ifs.fail()) 
	{
		return false;
	}
	int magic_number, num_items;

	ifs.read((char*) &magic_number, 4);
	ifs.read((char*) &num_items, 4);

	reverse_endian(&magic_number);
	reverse_endian(&num_items);

	for (size_t i = 0; i < num_items; i++) {
		unsigned char label;
		ifs.read((char*) &label, 1);
		labels->push_back((int) label);
	}
	return true;
}

// from tiny_cnn
struct mnist_header {
	int magic_number;
	int num_items;
	int num_rows;
	int num_cols;
};

// from tiny_cnn (kinda)
bool parse_mnist_images(const std::string& image_file, 
	std::vector<std::vector<float>> *images,
	float scale_min = -1.0, float scale_max = 1.0,
	int x_padding = 0, int y_padding = 0) 
{
	std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

	if (ifs.bad() || ifs.fail())
	{
			return false;
	}
	mnist_header header;

	// read header
	ifs.read((char*) &header.magic_number, 4);
	ifs.read((char*) &header.num_items, 4);
	ifs.read((char*) &header.num_rows, 4);
	ifs.read((char*) &header.num_cols, 4);

	reverse_endian(&header.magic_number);
	reverse_endian(&header.num_items);
	reverse_endian(&header.num_rows);
	reverse_endian(&header.num_cols);

		
	const int width = header.num_cols + 2 * x_padding;
	const int height = header.num_rows + 2 * y_padding;

	// read each image
	for (size_t i = 0; i < header.num_items; i++) 
	{
		std::vector<float> image;
		std::vector<unsigned char> image_vec(header.num_rows * header.num_cols);

		ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);
		image.resize(width * height, scale_min);
	
		for (size_t y = 0; y < header.num_rows; y++)
		{
			for (size_t x = 0; x < header.num_cols; x++)
				image[width * (y + y_padding) + x + x_padding] = 
					(image_vec[y * header.num_cols + x] / 255.0f) * (scale_max - scale_min) + scale_min;
		}
		
		images->push_back(image);
	}
	return true;
}

// == load data (MNIST-28x28x1 size, no padding, pixel range -1 to 1)
bool parse_test_data(std::string &data_path, std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels, 
	float min_val=-1.f, float max_val=1.f, int padx=0, int pady=0)
{
	if(!parse_mnist_images(data_path+"/t10k-images.idx3-ubyte", &test_images, min_val, max_val, padx, pady)) 
		if (!parse_mnist_images(data_path + "/t10k-images-idx3-ubyte", &test_images, min_val, max_val, padx, pady))
			return false;
	if(!parse_mnist_labels(data_path+"/t10k-labels.idx1-ubyte", &test_labels)) 
		if (!parse_mnist_labels(data_path + "/t10k-labels-idx1-ubyte", &test_labels)) return false;
	return true;
}
bool parse_train_data(std::string &data_path, std::vector<std::vector<float>> &train_images, std::vector<int> &train_labels, 
	float min_val=-1.f, float max_val=1.f, int padx=0, int pady=0)
{
	if(!parse_mnist_images(data_path+"/train-images.idx3-ubyte", &train_images, min_val, max_val, padx, pady))
		if (!parse_mnist_images(data_path + "/train-images-idx3-ubyte", &train_images, min_val, max_val, padx, pady))
			return false;
	if(!parse_mnist_labels(data_path+"/train-labels.idx1-ubyte", &train_labels))
		if (!parse_mnist_labels(data_path + "/train-labels-idx1-ubyte", &train_labels)) return false;
	return true;
}
}


