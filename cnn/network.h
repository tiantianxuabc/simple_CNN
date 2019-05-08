#pragma once

#include <string>
#include <iostream> 
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "layer.h"
#include "solver.h"
#include "activation.h"
#include "cost.h"
#include <opencv.hpp>






namespace Simple_cnn
{

#ifdef CNN_CV3
// forward declare these for data augmentation
cv::Mat matrix2cv(const Simple_cnn::matrix &m, bool uc8 = false);
Simple_cnn::matrix cv2matrix(cv::Mat &m);
Simple_cnn::matrix transform(const Simple_cnn::matrix in, const int x_center, const int y_center, int out_dim, float theta = 0, float scale = 1.f);
#endif


	void replace_str(std::string& str, const std::string& from, const std::string& to) 
	{
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos)
		{
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}


// returns Energy (euclidean distance / 2) and max index
float match_labels(const float *out, const float *target, const int size, int *best_index = NULL)
{
	float E = 0;
	int max_j = 0;
	for (int j = 0; j<size; j++)
	{
		E += (out[j] - target[j])*(out[j] - target[j]);
		if (out[max_j]<out[j]) 
			max_j = j;
	}
	if (best_index) 
		*best_index = max_j;
	E *= 0.5;
	return E;
}
// returns index of highest value (arg-max)
int arg_max(const float *out, const int size)
{
	int max_j = 0;
	for (int j = 0; j < size; j++)
	{
		if (out[max_j] < out[j])
		{
			max_j = j;
		}
	}
	return max_j;
}

//----------------------------------------------------------------------
//  network  
//  - class that holds all the layers and connection information
//	- runs forward prediction

class network
{
	
	int _size;  // output size	
	static const int MAIN_LAYER_SET = 0;

	// training related stuff
	int _batch_size;   // determines number of dW sets 
	float _skip_energy_level;
	bool _smart_train;
	std::vector <float> _running_E;
	double _running_sum_E;
	cost_function *_cost_function;
	solver *_solver;
	static const unsigned char BATCH_FREE = 0, BATCH_COMPLETE = 2;
	static const int BATCH_FILLED_COMPLETE = -2;

public:	
	// training progress stuff
	int train_correct;
	int train_skipped;
	int stuck_counter;
	int train_updates;
	int train_samples;
	int epoch_count;
	int max_epochs;
	float best_estimated_accuracy;
	int best_accuracy_count;
	float old_estimated_accuracy;
	float estimated_accuracy;
		
	
	// here we has a set of the layers to allow batch processing
	std::vector<base_layer *> layer_sets;
	
	std::map<std::string, int> layer_map;  // name-to-index of layer for layer management
	std::vector<std::pair<std::string, std::string>> layer_graph; // pairs of names of layers that are connected
	std::vector<matrix *> W; // these are the weights between/connecting layers 

	// these sets are needed because we need copies for each item in mini-batch
	std::vector< std::vector<matrix>> dW_sets; // only for training, will have _batch_size of these
	std::vector< std::vector<matrix>> dbias_sets; // only for training, will have _batch_size of these
	std::vector< unsigned char > batch_open; // only for training, will have _batch_size of these	
	

	network(const char* opt_name=NULL):_skip_energy_level(0.f), _batch_size(1) 
	{ 		
		_size=0;  
		_solver = new_solver(opt_name);
		_cost_function = NULL;
		
		
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size);
		_running_sum_E = 0.;
		train_correct = 0;
		train_samples = 0;
		train_skipped = 0;
		epoch_count = 0; 
		max_epochs = 1000;
		train_updates = 0;
		estimated_accuracy = 0;
		old_estimated_accuracy = 0;
		stuck_counter = 0;
		best_estimated_accuracy=0;
		best_accuracy_count=0;	
	
	}
	
	~network() 
	{
		clear();
		if (_cost_function) delete _cost_function;
		if(_solver) delete _solver; 
		
	}

	// call clear if you want to load a different configuration/model
	void clear()
	{
		for (auto l : layer_sets)
			delete l;
		layer_sets.clear();
		for(auto w : W) if(w) delete w;  
		W.clear();
		layer_map.clear();
		layer_graph.clear();
	}

	// output size of final layer;
	int out_size() 
	{
		return _size;
	}

	// get input size 
	bool get_input_size(int *w, int *h, int *c)
	{
		if(layer_sets.size()<1)
			return false; 
		*w=layer_sets[0]->node.cols;
		*h=layer_sets[0]->node.rows;
		*c=layer_sets[0]->node.chans;
		return true;
	}

	// used to add some noise to weights
	void heat_weights()
	{
		for(auto w : W)
		{
			if (!w) continue;
			matrix noise(w->cols, w->rows, w->chans);
			noise.fill_random_normal(1.f/ noise.size());
			*w += noise; 
		}
	}

	// used to add some noise to weights
	void remove_means()
	{
		for(auto w : W)
			if(w) w->remove_mean();
	}

	// used to push a layer back in the ORDERED list of layers
	// if connect_all() is used, then the order of the push_back is used to connect the layers
	// when forward or backward propagation, this order is used for the serialized order of calculations 
	// Layer_name must be unique.
	bool push_back(const char *layer_name, const char *layer_config)
	{
		if(layer_map[layer_name]) return false; //already exists
		base_layer *layer = new_layer(layer_name, layer_config);		

		layer_map[layer_name] = (int)layer_sets.size();
		layer_sets.push_back(layer);
		_size = layer->fan_size(); //return node.chans*node.rows*node.cols;
		
		return true;
	}

	// connect 2 layers together and initialize weights
	// top and bottom concepts are reversed from literature
	// my 'top' is the input of a forward() pass and the 'bottom' is the output
	// perhaps 'top' traditionally comes from the brain model, but my 'top' comes
	// from reading order (information flows top to bottom)
	void connect(const char *layer_name_top, const char *layer_name_bottom) 
	{
		size_t i_top=layer_map[layer_name_top];
		size_t i_bottom=layer_map[layer_name_bottom];

		base_layer *l_top= layer_sets[i_top];
		base_layer *l_bottom= layer_sets[i_bottom];
		
		int w_i=(int)W.size();
		matrix *w = l_bottom->new_connection(*l_top, w_i);
		W.push_back(w);
		layer_graph.push_back(std::make_pair(layer_name_top,layer_name_bottom));
	

		// we need to let solver prepare space for stateful information 
		if (_solver)
		{
			if (w)
				_solver->push_back(w->cols, w->rows, w->chans);
			else
				_solver->push_back(1, 1, 1);
		}

		int fan_in=l_bottom->fan_size();
		int fan_out=l_top->fan_size();

		// ToDo: this may be broke when 2 layers connect to one. need to fix (i.e. resnet)
		// after all connections, run through and do weights with correct fan count

		// initialize weights - ToDo: separate and allow users to configure(?)
		if (w && l_bottom->has_weights())
		{
			if (strcmp(l_bottom->p_act->name, "tanh") == 0)
			{
				// xavier : for tanh
				float weight_base = (float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				w->fill_random_uniform(weight_base);
			}
			else if (strcmp(l_bottom->p_act->name, "sigmoid") == 0) 
			{
				// xavier : for sigmoid
				float weight_base = 4.f*(float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				w->fill_random_uniform(weight_base);
			}
			else if ((strcmp(l_bottom->p_act->name, "lrelu") == 0) || (strcmp(l_bottom->p_act->name, "relu") == 0)
				|| (strcmp(l_bottom->p_act->name, "vlrelu") == 0) || (strcmp(l_bottom->p_act->name, "elu") == 0))
			{
				// he : for relu
				float weight_base = (float)(std::sqrt(2. / (double)fan_in));
				w->fill_random_normal(weight_base);
			}
			else
			{
				// lecun : orig
				float weight_base = (float)(std::sqrt(1. / (double)fan_in));
				w->fill_random_uniform(weight_base);
			}
		}
		else 
			if (w)
				w->fill(0);
	}

	// automatically connect all layers in the order they were provided 
	// easy way to go, but can't deal with branch/highway/resnet/inception types of architectures
	void connect_all()
	{	
		for (int j = 0; j < (int)layer_sets.size() - 1; j++)
		{
			connect(layer_sets[j]->name.c_str(), layer_sets[j + 1]->name.c_str());
		}		
	}

	int get_layer_index(const char *name)
	{
		for (int j = 0; j < (int)layer_sets.size(); j++)
		{
			if (layer_sets[j]->name.compare(name) == 0)
			{
				return j;
			}
		}
		return -1;
	}

	// get the list of layers used (but not connection information)
	std::string get_configuration()
	{
		std::string str;
		for (int j = 0; j < (int)layer_sets.size(); j++)
		{
			str += "  " + std::to_string((long long)j) + " : " + layer_sets[j]->name + " : " + layer_sets[j]->get_config_string();
		}
		str += "\n";
		// print layer links
		if (layer_graph.size() <= 0) 
			return str;
		
		for (int j = 0; j < (int)layer_graph.size(); j++)
		{
			if (j % 3 == 0) 
				str += "  ";
			if((j % 3 == 1)|| (j % 3 == 2)) 
				str += ", ";
			str += layer_graph[j].first + "-" + layer_graph[j].second;
			if (j % 3 == 2) str += "\n";
		}
		return str;
	}

	// performs forward pass and returns class index
	// do not delete or modify the returned pointer. it is a live pointer to the last layer in the network
	int predict_class(const float *in)
	{
		const float* out = forward(in);
		return arg_max(out, out_size());
	}

	//----------------------------------------------------------------------------------------------------------
	// F O R W A R D
	//
	// the main forward pass 
	float* forward(const float *in, int _train=0)
	{		
		std::vector<base_layer *> inputs;
		for(auto layer : layer_sets)
		{
			if (dynamic_cast<input_layer*> (layer) != NULL)
			{
				inputs.push_back(layer);
			}
			
			layer->node.fill(0.f);
		}
		// first layer assumed input. copy input to it 
		const float *in_ptr = in;				
		for(auto layer : inputs)
		{
			memcpy(layer->node.x, in_ptr, sizeof(float)*layer->node.size());
			in_ptr += layer->node.size();
		}
		

		// for all layers
		for(auto layer : layer_sets)
		{
			// add bias and activate these outputs (they should all be summed up from other branches at this point)
			layer->activate_nodes(); 
			
			//for(int j=0; j<layer->node.chans; j++) for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";
			// send output signal downstream (note in this code 'top' is input layer, 'bottom' is output - bucking tradition
			for (auto &link : layer->forward_linked_layers)
			{
				// instead of having a list of paired connections, just use the shape of W to determine connections
				int connection_index = link.first; 
				base_layer *p_bottom = link.second;


				// weight distribution of the signal to layers under it
				p_bottom->accumulate_signal(*layer, *W[connection_index], _train);		
			}
		}

		return layer_sets[layer_sets.size()-1]->node.x;
	}

	//----------------------------------------------------------------------------------------------------------
	// W R I T E
	//
	// write parameters to stream/file
	// note that this does not persist intermediate training information that could be needed to 'pickup where you left off'
	bool write(std::ofstream& ofs, bool binary = false)
	{
		// save layers
		int layer_cnt = (int)layer_sets.size();

		ofs<<"cnn01" << std::endl;
		ofs<<(int)(layer_cnt)<<std::endl;
		
		for (int j = 0; j < (int)layer_sets.size(); j++)
		{
			ofs << layer_sets[j]->name << std::endl << layer_sets[j]->get_config_string();
		}


		// save graph
		ofs<<(int)layer_graph.size()<<std::endl;
		for (int j = 0; j < (int)layer_graph.size(); j++)
		{
			ofs << layer_graph[j].first << std::endl << layer_graph[j].second << std::endl;
		}

		if(binary)
		{
			ofs<<(int)1<<std::endl; // flags that this is binary data
			// binary version to save space if needed
			// save bias info
			for (int j = 0; j < (int)layer_sets.size(); j++)
			{
				if (layer_sets[j]->use_bias())
					ofs.write((char*)layer_sets[j]->bias.x, layer_sets[j]->bias.size() * sizeof(float));
			}
			// save weights
			for (int j = 0; j < (int)W.size(); j++)
			{
				if (W[j])
					ofs.write((char*)W[j]->x, W[j]->size()*sizeof(float));
			}
		}
		else
		{
			ofs<<(int)0<<std::endl;
			// save bias info
			for(int j=0; j<(int)layer_sets.size(); j++)
			{
				if (layer_sets[j]->use_bias())
				{
					for (int k = 0; k < layer_sets[j]->bias.size(); k++)  ofs << layer_sets[j]->bias.x[k] << " ";
					ofs << std::endl;
				}
			}
			// save weights
			for(int j=0; j<(int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) ofs << W[j]->x[i] << " ";
					ofs << std::endl;
				}
			}
		}
		ofs.flush();
		
		return true;
	}

	bool write(std::string &filename, bool binary = false, bool final = false)
	{ 
		std::ofstream temp((const char *)filename.c_str(), std::ios::binary);
		return write(temp, binary);
	}

	bool write(char *filename, bool binary = false, bool final = false) 
	{
		std::string str= filename;
		return write(str, binary, final); 
	}

	// read network from a file/stream
	
	std::string getcleanline(std::istream& ifs)
	{
		std::string s;	

		std::istream::sentry se(ifs, true);
		std::streambuf* sb = ifs.rdbuf();

		for (;;) {
			int c = sb->sbumpc();
			switch (c) {
			case '\n':
				return s;
			case '\r':
				if (sb->sgetc() == '\n') sb->sbumpc();
				return s;
			case EOF:
				// Also handle the case when the last line has no line ending
				if (s.empty()) ifs.setstate(std::ios::eofbit);
				return s;
			default:
				s += (char)c;
			}
		}
	}
	
	//----------------------------------------------------------------------------------------------------------
	// R E A D
	//
	bool read(std::istream &ifs)
	{
		if (!ifs.good()) return false;
		std::string s;
		s = getcleanline(ifs);
		int layer_count;
		int version = 0;
		if (s.compare("cnn01") == 0)
		{
			s = getcleanline(ifs);
			layer_count = atoi(s.c_str());
			version = 1;
		}
		else if (s.compare("cnn:") == 0)
		{
			version = -1;
			int cnt = 1;

			while (!ifs.eof())
			{
				s = getcleanline(ifs);
				if (s.empty()) continue;
				push_back(int2str(cnt).c_str(), s.c_str());
				cnt++;
			}
			connect_all();

			// copies batch=0 stuff to other batches
			///sync_layer_sets();
			return true;
		}
		else
			layer_count = atoi(s.c_str());
		// read layer def
		std::string layer_name;
		std::string layer_def;
		for (auto i = 0; i < layer_count; i++)
		{
			layer_name = getcleanline(ifs);
			layer_def = getcleanline(ifs);
			push_back(layer_name.c_str(), layer_def.c_str());
		}

		// read graph
		int graph_count;
		ifs >> graph_count;
		getline(ifs, s); // get endline
		if (graph_count <= 0)
		{
			connect_all();
		}
		else
		{
			std::string layer_name1;
			std::string layer_name2;
			for (auto i = 0; i < graph_count; i++)
			{
				layer_name1 = getcleanline(ifs);
				layer_name2 = getcleanline(ifs);
				connect(layer_name1.c_str(), layer_name2.c_str());
			}
		}

		int binary;
		s = getcleanline(ifs); // get endline
		binary = atoi(s.c_str());

		// binary version to save space if needed
		if (binary == 1)
		{
			for (int j = 0; j < (int)layer_sets.size(); j++)
				if (layer_sets[j]->use_bias())
				{
					ifs.read((char*)layer_sets[j]->bias.x, layer_sets[j]->bias.size() * sizeof(float));
				}
			for (int j = 0; j < (int)W.size(); j++)
			{

				if (W[j])
				{
					ifs.read((char*)W[j]->x, W[j]->size() * sizeof(float));
				}
			}
		}
		else if (binary == 0)// text version
		{
			// read bias
			for (int j = 0; j < layer_count; j++)
			{
				if (layer_sets[j]->use_bias())
				{

					for (int k = 0; k < layer_sets[j]->bias.size(); k++)
					{
						ifs >> layer_sets[j]->bias.x[k];
					}
					ifs.ignore();// getline(ifs, s); // get endline
				}
			}

			// read weights
			for (auto j = 0; j < (int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) ifs >> W[j]->x[i];
					ifs.ignore(); //getline(ifs, s); // get endline
				}
			}
		}
		return true;
	}

	bool read(std::string filename)
	{
		std::ifstream fs(filename.c_str(),std::ios::binary);
		if (fs.is_open())
		{
			bool ret = read(fs);
			fs.close();
			return ret;
		}
		else return false;
	}

	bool read(const char *filename)
	{ 
		return  read(std::string(filename)); 
	}

#ifndef CNN_NO_TRAINING  // this is surely broke by now and will need to be fixed

	// ===========================================================================
	// training part
	// ===========================================================================

	// resets the state of all batches to 'free' state
	void reset_mini_batch() 
	{
		memset(batch_open.data(), BATCH_FREE, batch_open.size());
	}
	
	// sets up number of mini batches (storage for sets of weight deltas)
	void set_mini_batch_size(int batch_cnt)
	{
		if (batch_cnt<1) batch_cnt = 1;
		_batch_size = batch_cnt;
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size); 
		reset_mini_batch();
	}
	
	int get_mini_batch_size() 
	{ 
		return _batch_size;
	}

	// return index of next free batch
	// or returns -2 (BATCH_FILLED_COMPLETE) if no free batches - all complete (need a sync call)
	// or returns -1 (BATCH_FILLED_IN_PROCESS) if no free batches - some still in progress (must wait to see if one frees)
	int get_next_open_batch()
	{	
		int filled = 0;
		for (int i = 0; i<batch_open.size(); i++)
		{
			if (batch_open[i] == BATCH_FREE)
				return i;			
			if (batch_open[i] == BATCH_COMPLETE)
				filled++;
		}		
		if (filled == batch_open.size())
			return BATCH_FILLED_COMPLETE; // all filled and complete	
	}

	//----------------------------------------------------------------------------------------------------------
	// s y n c   m i n i   b a t c h
	//
	// apply all weights to first set of dW, then apply to model weights 
	void sync_mini_batch()
	{
	
		int layer_cnt = (int)layer_sets.size();
		base_layer *layer;

		 // sum contributions 
		for (int k = layer_cnt - 1; k >= 0; k--)
		{
			layer = layer_sets[k];
			//std::cout << "layer name: " << layer->name << std::endl;
			for(auto &link : layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				if (batch_open[0] == BATCH_FREE)
					dW_sets[MAIN_LAYER_SET][w_index].fill(0);
				for (int b = 1; b< _batch_size; b++)
				{
					if (batch_open[b] == BATCH_COMPLETE) 
						dW_sets[MAIN_LAYER_SET][w_index] += dW_sets[b][w_index];
				}				
			}
			if (dynamic_cast<convolution_layer*> (layer) != NULL)
			{	
				continue;
			}

			
			// bias stuff... that needs to be fixed for conv layers perhaps
			if (batch_open[0] == BATCH_FREE)
				dbias_sets[MAIN_LAYER_SET][k].fill(0);
			for (int b = 1; b< _batch_size; b++)
			{
				if (batch_open[b] == BATCH_COMPLETE) 
					dbias_sets[MAIN_LAYER_SET][k] += dbias_sets[b][k];
			}
		}

		// update weights
		for (int k = layer_cnt - 1; k >= 0; k--)
		{
			layer = layer_sets[k];
			
			for(auto &link : layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				if (dW_sets[MAIN_LAYER_SET][w_index].size() > 0)
				{
					if (W[w_index]) 
					{
						////std::cout << "has weight" << std::endl;
						_solver->increment_w(W[w_index], w_index, dW_sets[MAIN_LAYER_SET][w_index]);
					}
				}

			}
			//std::cout << "update bias " << layer_sets[k]->name << std::endl;
			layer->update_bias(dbias_sets[MAIN_LAYER_SET][k], _solver->learning_rate);
		}	
		// prepare to start mini batch over
		reset_mini_batch();
		train_updates++; // could have no updates .. so this is not exact
	}


	float get_learning_rate() 
	{
		if(!_solver) 
			bail("set solver");
		return _solver->learning_rate;
	}
	void set_learning_rate(float alpha)
	{
		if(!_solver) bail("set solver"); 
		_solver->learning_rate=alpha;
	}
	void reset_solver() 
	{
		if (!_solver)
			bail("set solver");
		_solver->reset();
	}
	bool get_smart_training()
	{
		return _smart_train;
	}
	void set_smart_training(bool _use_train) 
	{ 
		_smart_train = _use_train;
	}
	float get_smart_train_level()
	{
		return _skip_energy_level;
	}
	void set_smart_train_level(float _level)
	{ 
		_skip_energy_level = _level;
	}
	void set_max_epochs(int max_e)
	{ 
		if (max_e <= 0) 
			max_e = 1; 
		max_epochs = max_e; 
	}
	int get_epoch() 
	{
		return epoch_count; 
	}

// ===========================================================================
// training part
// ===========================================================================

		
	// call before starting training for current epoch
	void start_epoch(std::string loss_function="mse")
	{
		_cost_function=new_cost_function(loss_function);
		train_correct = 0;
		train_skipped = 0;
		train_updates = 0;
		train_samples = 0;
		if (epoch_count == 0)
			reset_solver();
	
		// accuracy not improving .. slow learning
		if(_smart_train &&  best_accuracy_count > 4)
		{
			stuck_counter++;
			set_learning_rate((0.5f)*get_learning_rate());
			if (get_learning_rate() < 0.000001f)
			{
				heat_weights();
				set_learning_rate(0.000001f);
				stuck_counter++;// end of the line.. so speed up end
			}
			best_accuracy_count = 0;
		}

		old_estimated_accuracy = estimated_accuracy;
		estimated_accuracy = 0;
		//_skip_energy_level = 0.05;
		_running_sum_E = 0;
	}
	
	// time to stop?
	bool evalu_left_the_building()
	{
		// 2 stuck x 4 non best accuracy to quit = 8 times no improvement 
		if ((epoch_count>max_epochs) || (stuck_counter > 3)) 
			return true;
		else
			return false;
	}

	// call after putting all training samples through this epoch
	bool end_epoch()
	{
		// run leftovers through mini-batch
		sync_mini_batch();
		epoch_count++;

		// estimate accuracy of validation run 
		estimated_accuracy = 100.f*train_correct / train_samples;

		if (train_correct > best_estimated_accuracy)
		{
			best_estimated_accuracy = (float)train_correct;
			best_accuracy_count = 0;
			stuck_counter = 0;
		}
		else 
			best_accuracy_count++;

		return evalu_left_the_building();
	}

	// if smart training was thinking about exiting, calling reset will make it think everything is OK
	void reset_smart_training()
	{
		stuck_counter=0;
		best_accuracy_count = 0;
		best_estimated_accuracy = 0;
	}

	//----------------------------------------------------------------------------------------------------------
	// u p d a t e _ s m a r t _ t r a i n
	//
	void update_smart_train(const float E, bool correct)
	{
		train_samples++;
		if (correct) 
			train_correct++;

		if (_smart_train)
		{
			_running_E.push_back(E);
			_running_sum_E += E;
			const int SMART_TRAIN_SAMPLE_SIZE = 1000;

			int s = (int)_running_E.size();
			if (s >= SMART_TRAIN_SAMPLE_SIZE)
			{
				_running_sum_E /= (double)s;
				std::sort(_running_E.begin(), _running_E.end());
				float top_fraction = (float)_running_sum_E*10.f; //10.
				const float max_fraction = 0.75f;
				const float min_fraction = 0.075f;// 0.03f;

				if (top_fraction > max_fraction) top_fraction = max_fraction;
				if (top_fraction < min_fraction) top_fraction = min_fraction;
				int index = s - 1 - (int)(top_fraction*(s - 1));

				if (_running_E[index] > 0) 
					_skip_energy_level = _running_E[index];

				_running_sum_E = 0;

				_running_E.clear();
			}
		}
		if (E > 0 && E < _skip_energy_level)
		{
			train_skipped++;
		}
	}
	// finish back propagation through the hidden layers
	void backward_hidden(const int my_batch_index)
	{		
		const int layer_cnt = (int)layer_sets.size();
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer;

		// update hidden layers
		// start at lower layer and push information up to previous layer
		for (int k = last_layer_index; k >= 0; k--)
		{			
			layer = layer_sets[k];
			// all the signals should be summed up to this layer by now, so we go through and take the grad of activiation
			int nodes = layer->node.size();
			// already did last layer, so skip it
			if (k < last_layer_index)
			{
				for (int i = 0; i < nodes; i++)
				{
					layer->delta.x[i] *= layer->df(layer->node.x, i, nodes);
				}
			}

			// now pass that signal upstream
			for(auto &link : layer->backward_linked_layers) // --- 50% of time this loop
			{
				base_layer *p_top = link.second;
				// note all the delta[connections[i].second] should have been calculated by time we get here
				layer->distribute_delta(*p_top, *W[link.first]);
			}
		}

		// update weights - shouldn't matter the direction we update these 
		// we can stay in backwards direction...
		// it was not faster to combine distribute_delta and increment_w into the same loop
		int size_W = (int)W.size();
		dW_sets[my_batch_index].resize(size_W);
		dbias_sets[my_batch_index].resize(layer_cnt);
		for (int k = last_layer_index; k >= 0; k--)
		{
			layer = layer_sets[k];			
			for(auto &link : layer->backward_linked_layers)
			{
				base_layer *p_top = link.second;
				int w_index = (int)link.first;
				layer->calculate_dw(*p_top, dW_sets[my_batch_index][w_index]);
			}
			if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

			dbias_sets[my_batch_index][k] = layer->delta;
		}
		// if all batches finished, update weights
		
		batch_open[my_batch_index] = BATCH_COMPLETE;
		int next_index = get_next_open_batch();
		if (next_index == BATCH_FILLED_COMPLETE) // all complete
			sync_mini_batch(); // resets _batch_index to 0
		
	}

	
	//----------------------------------------------------------------------------------------------------------
	// T R A I N   C L A S S 
	//
	// after starting epoch, call this to train against a class label
	// label_index must be 0 to out_size()-1

	bool train_class(float *in, int label_index)
	{
		float *input = in;
	
		// get next free mini_batch slot		
		int my_batch_index = get_next_open_batch();
		
		if (my_batch_index < 0)
			return false;

		// run through forward to get nodes activated
		forward(input, 1);
		
		// set all deltas to zero
		//std::cout << "initiation delta of every layer as 0" << std::endl;
		for (auto layer : layer_sets)
		{
			layer->delta.fill(0.f);
		}

		int layer_cnt = (int)layer_sets.size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer = layer_sets[last_layer_index];
		const int layer_node_size = layer->node.size();
		const int layer_delta_size = layer->delta.size();

		
		float E = 0;
		int max_j_out = 0;
		int max_j_target = label_index;

		// was passing this in, but may as well just create it on the fly
		// a vector mapping the label index to the desired target output node values
		// all -1 except target node 1
		std::vector<float> target;
		if((std::string("sigmoid").compare(layer->p_act->name) == 0) || (std::string("softmax").compare(layer->p_act->name) == 0))
			target = std::vector<float>(layer_node_size, 0);
		else
			target = std::vector<float>(layer_node_size, -1);
		
		if(label_index>=0 && label_index<layer_node_size)
			target[label_index] = 1;

		//const float grad_fudge = 1.0f;
		// because of numerator/demoninator cancellations which prevent a divide by zero issue, 
		// we need to handle some things special on output layer
		float cost_activation_type = 0;
		if ((std::string("sigmoid").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0)) 
			cost_activation_type = 1;
		else if ((std::string("softmax").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 1;
		else if ((std::string("tanh").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0)) 
			cost_activation_type = 4;
	
		
		for (int j = 0; j < layer_node_size; j++)
		{
			if(cost_activation_type>0)
				layer->delta.x[j] = cost_activation_type*(layer->node.x[j]- target[j]);
			else
				layer->delta.x[j] = _cost_function->d_cost(layer->node.x[j], target[j])*layer->df(layer->node.x, j, layer_node_size);

			// pick best response
			if (layer->node.x[max_j_out] < layer->node.x[j]) max_j_out = j;
			// for better E maybe just look at 2 highest scores so zeros don't dominate 

			float f= mse::cost(layer->node.x[j], target[j]);
			E += f;
		}
	
		E /= (float)layer_node_size;
		// check for NAN
		if (E != E) bail("network blew up - try lowering learning rate\n");
		
		// critical section in here, blocking update
		bool match = false;
		if ((max_j_target == max_j_out)) 
			match = true;
		update_smart_train(E, match);

		if (E>0 && E<_skip_energy_level && _smart_train && match)
		{
			
			batch_open[my_batch_index] = BATCH_FREE;
			
			return false;  // return without doing training
		}
		backward_hidden(my_batch_index);
		return true;
	}
	
	
#else

	float get_learning_rate() {return 0;}
	void set_learning_rate(float alpha) {}
	void train(float *in, float *target){}
	void reset() {}
	float get_smart_train_level() {return 0;}
	void set_smart_train_level(float _level) {}
	bool get_smart_train() { return false; }
	void set_smart_train(bool _use) {}

#endif

};

}