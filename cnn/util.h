#pragma once


#include <time.h>
#include <string>

#include <chrono>

#include "core_math.h"
#include "network.h"


#include <opencv.hpp>

#if DEBUG
#pragma comment(lib, "opencv_world400d")
#else
#pragma comment(lib, "opencv_world400")
#endif



namespace Simple_cnn
{

// class to handle timing and drawing text progress output
class progress
{

public:
	progress(int size=-1, const char *label=NULL ) {reset(size, label);}


	std::chrono::time_point<std::chrono::system_clock>  start_progress_time;

	unsigned int total_progress_items;
	std::string label_progress;
	// if default values used, the values won't be changed from last call
	void reset(int size=-1, const char *label=NULL ) 
	{

		start_progress_time= std::chrono::system_clock::now();

		if(size>0) total_progress_items=size; if(label!=NULL) label_progress=label;
	}
	float elapsed_seconds() 
	{	
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_progress_time);
		return (float)time_span.count();

	}
	float remaining_seconds(int item_index)
	{
		float elapsed_dt = elapsed_seconds();
		float percent_complete = 100.f*item_index/total_progress_items;
		if(percent_complete>0) return ((elapsed_dt/percent_complete*100.f)-elapsed_dt);
		return 0.f;
	}
	// this doesn't work correctly with g++/Cygwin
	// the carriage return seems to delete the text... 
	void draw_progress(int item_index)
	{
		int time_remaining = (int)remaining_seconds(item_index);
		float percent_complete = 100.f*item_index/total_progress_items;
		if (percent_complete > 0)
		{
			std::cout << label_progress << (int)percent_complete << "% (" << (int)time_remaining << "sec remaining)              \r"<<std::flush;
		}
	}
	void draw_header(std::string name, bool _time=false)
	{
		std::string header = "==  " + name + "  ";

		int seconds = 0;
		std::string elapsed;
		int L = 79 - (int)header.length();
		if (_time)
		{
			seconds = (int)elapsed_seconds();
			int minutes = (int)(seconds / 60);
			int hours = (int)(minutes / 60);
			seconds = seconds - minutes * 60;
			minutes = minutes - hours * 60;
			std::string min_string = std::to_string((long long)minutes);
			if (min_string.length() < 2) min_string = "0" + min_string;
			std::string sec_string = std::to_string((long long)seconds);
			if (sec_string.length() < 2) sec_string = "0" + sec_string;
			elapsed = " " + std::to_string((long long)hours) + ":" + min_string + ":" + sec_string;
			L-= (int)elapsed.length();
		}
		for (int i = 0; i<L; i++) header += "=";
		if (_time)
			std::cout << header << elapsed << std::endl;
		else 
			std::cout << header << std::endl;
	}
};

class html_log
{
	struct log_stuff
	{
		std::string str;
		float test_accurracy;
		float train_accurracy_est;
	};
	std::vector <log_stuff> log;
	std::string header;
	std::string notes;
public:
	html_log() {};

	// the header you set here should have tab \t separated column headers that match what will go in the row
	// the first 3 columns are always epoch, test accuracy, est accuracy
	void set_table_header(std::string tab_header) { header=tab_header;}
	// tab_row should be \t separated things to put after first 3 columns
	void add_table_row(float train_acccuracy, float test_accuracy, std::string tab_row)
	{
		log_stuff s;
		s.str = tab_row; s.test_accurracy = test_accuracy; s.train_accurracy_est = train_acccuracy;
		log.push_back(s);
	}
	void set_note(std::string msg) {notes = msg;}
	bool write(std::string filename) {

		std::string top = "<!DOCTYPE html><html><head><meta http-equiv=\"content - type\" content=\"text/html; charset = UTF - 8\"><style>table, th, td{border: 1px solid black; border - collapse: collapse; } th, td{ padding: 5px;}</style><meta name=\"robots\" content=\"noindex, nofollow\"><meta name=\"googlebot\" content=\"noindex, nofollow\"><meta http-equiv=\"refresh\" content=\"30\"/><script type=\"text/javascript\" src=\"/js/lib/dummy.js\"></script><link rel=\"stylesheet\" type=\"text/css\" href=\"/css/result-light.css\"><style type=\"text/css\"></style><title>Simple_cnn CNN Training Report</title></head><body><script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script><b><center>Training Summary <script type=\"text/javascript\">document.write(Date());</script></center></b><div id = \"chart_div\"></div><script type = 'text/javascript'>//<![CDATA[\n";
		top += "google.charts.load('current', { packages: ['corechart', 'line'] });\ngoogle.charts.setOnLoadCallback(drawLineColors);\nfunction drawLineColors() {";
		top += "\nvar data = new google.visualization.DataTable();data.addColumn('number', 'Epoch');data.addColumn('number', 'Training Estimate');data.addColumn('number', 'Validation Testing');data.addRows([";
		std::string data = "";
		float min = 100;
		float max_10 = 0;
		for (int i = 0; i < log.size(); i++)
		{
			if ((100.f - log[i].train_accurracy_est) < min) min = (100.f - log[i].train_accurracy_est);
			if ((100.f - log[i].test_accurracy) < min) min = (100.f - log[i].test_accurracy);
			if ((100.f - log[i].train_accurracy_est) > max_10) max_10 = (100.f - log[i].train_accurracy_est);
			if ((100.f - log[i].test_accurracy) > max_10) max_10 = (100.f - log[i].test_accurracy);

			data += "[" + int2str(i) + "," + float2str(100.f - log[i].train_accurracy_est) + "," + float2str(100.f - log[i].test_accurracy) + "],";
		}
		float min_10 = min;
//		while (min_10 > min) min_10 /= 10.f;

		std::string mid = "]);var options = { 'height':400, hAxis: {title: 'Epoch', logScale: true},vAxis : {title: 'Error (%)', logScale: true, viewWindow: {min:"+float2str(min_10)+",max:"+ float2str(max_10)+"},ticks: [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2, 5, 10, 20, 50, 100] },colors : ['#313055','#F52B00'] };var chart = new google.visualization.LineChart(document.getElementById('chart_div')); chart.draw(data, options);}//]]>\n </script>";

		std::string msg = "<table style='width:100 %' align='center'>";
		int N = (int)log.size();
		msg += "<tr><td>" + header + "</td></tr>";
		int best = N - 1;
		int best_est = N - 1;
		for (int i = N - 1; i >= 0; i--)
		{
			if (log[i].test_accurracy > log[best].test_accurracy) best = i;
			if (log[i].train_accurracy_est > log[best_est].train_accurracy_est) best_est = i;
		}
		for (int i = N - 1; i >= 0; i--)
		{
			msg += "<tr><td>" + int2str(i + 1);
			// make best green
			if (i == best)	msg += "</td><td bgcolor='#00FF00'>";
			else msg += "</td><td>";
			msg+=float2str(log[i].test_accurracy);
			// mark bad trend in training
			if (i > best_est) msg += "</td><td bgcolor='#FFFF00'>";
			else msg += "</td><td>";
			msg+=float2str(log[i].train_accurracy_est)+ "</td><td>" + log[i].str + "</td></tr>";
		}
		replace_str(msg, "\t", "</td><td>");

		replace_str(notes, "\n", "<br>");
		std::string bottom = "</tr></table><br>"+notes+"</body></html>";

		std::ofstream f(filename.c_str());
		f << top; f << data; f << mid; f << msg; f << bottom;

		f.close();
		return true;
	}

};

#ifdef CNN_CV3


	// transforms image. 
	// x_center, y_center of input
	// out dim is size of output w or h
	// theta in degrees
	cv::Mat matrix2cv(const Simple_cnn::matrix &m, bool uc8)// = false)
	{
		cv::Mat cv_m;
		if (m.chans != 3)
		{
			cv_m = cv::Mat(m.cols, m.rows, CV_32FC1, m.x);
		}
		if (m.chans == 3)
		{
			cv::Mat in[3];
			in[0] = cv::Mat(m.cols, m.rows, CV_32FC1, m.x);
			in[1] = cv::Mat(m.cols, m.rows, CV_32FC1, &m.x[m.cols*m.rows]);
			in[2] = cv::Mat(m.cols, m.rows, CV_32FC1, &m.x[2 * m.cols*m.rows]);
			cv::merge(in, 3, cv_m);
		}
		if (uc8)
		{
			double min_, max_;
			cv_m = cv_m.reshape(1);
			cv::minMaxIdx(cv_m, &min_, &max_);
			cv_m = cv_m - min_;
			max_ = max_ - min_;
			cv_m /= max_;
			cv_m *= 255;
			cv_m = cv_m.reshape(m.chans, m.rows);
			if (m.chans != 3)
				cv_m.convertTo(cv_m, CV_8UC1);
			else
				cv_m.convertTo(cv_m, CV_8UC3);
		}
		return cv_m;
	}

	Simple_cnn::matrix cv2matrix(cv::Mat &m)
	{
		if (m.type() == CV_8UC1)
		{
			m.convertTo(m, CV_32FC1);
			m = m / 255.;
		}
		if (m.type() == CV_8UC3)
		{
			m.convertTo(m, CV_32FC3);
		}
		if (m.type() == CV_32FC1)
		{
			return Simple_cnn::matrix(m.cols, m.rows, 1, (float*)m.data);
		}
		if (m.type() == CV_32FC3)
		{
			cv::Mat in[3];
			cv::split(m, in);
			Simple_cnn::matrix out(m.cols, m.rows, 3);
			memcpy(out.x, in[0].data, m.cols*m.rows * sizeof(float));
			memcpy(&out.x[m.cols*m.rows], in[1].data, m.cols*m.rows * sizeof(float));
			memcpy(&out.x[2 * m.cols*m.rows], in[2].data, m.cols*m.rows * sizeof(float));
			return out;
		}
		return  Simple_cnn::matrix(0, 0, 0);
	}
	Simple_cnn::matrix transform(const Simple_cnn::matrix in, const int x_center, const int y_center,
		int out_dim, float theta, float scale)// theta =0 scale= 1.f)
	{
		const double _pi = 3.14159265358979323846;
		float cos_theta = (float)std::cos(theta / 180.*_pi);
		float sin_theta = (float)std::sin(theta / 180.*_pi);
		float half_dim = 0.5f*(float)out_dim / scale;

		cv::Point2f  pts1[4], pts2[4];
		pts1[0] = cv::Point2f(x_center - half_dim, y_center - half_dim);
		pts1[1] = cv::Point2f(x_center + half_dim, y_center - half_dim);
		pts1[2] = cv::Point2f(x_center + half_dim, y_center + half_dim);
		pts1[3] = cv::Point2f(x_center - half_dim, y_center + half_dim);

		pts2[0] = cv::Point2f(-half_dim, -half_dim);
		pts2[1] = cv::Point2f(half_dim, -half_dim);
		pts2[2] = cv::Point2f(half_dim, half_dim);
		pts2[3] = cv::Point2f(-half_dim, half_dim);

		// rotate around center spot
		for (int pt = 0; pt<4; pt++)
		{
			float x_t = (pts2[pt].x)*scale;
			float y_t = (pts2[pt].y)*scale;
			float x = cos_theta*x_t - sin_theta*y_t;
			float y = sin_theta*x_t + cos_theta*y_t;

			pts2[pt].x = x + (float)x_center;
			pts2[pt].y = y + (float)y_center;

			// we want to control how data is scaled down
			//		if (scale>1)
			//		{
			//			pts1[pt].x = pts1[pt].x / (float)scale;
			//			pts1[pt].y = pts1[pt].y / (float)scale;
			//		}
		}
		
		cv::Mat input = Simple_cnn::matrix2cv(in,false);

		//	if (scale>1)
		//		cv::resize(in, input, cv::Size(0, 0), 1. / scale, 1. / scale);
		//	else
		//		input = in;


		cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);
		cv::Mat cv_out;

		cv::warpPerspective(input, cv_out,
			cv::getPerspectiveTransform(pts1, pts2),
			cv::Size((int)((float)out_dim), (int)((float)out_dim)),
			cv::INTER_AREA, cv::BORDER_REPLICATE); //cv::INTER_LINEAR

												   //INTER_AREA


												   //	double min;
												   //	cv::minMaxIdx(cv_out, &min);
												   //	std::cout << "min: " << min << "||";
		return Simple_cnn::cv2matrix(cv_out);
	}


void save(Simple_cnn::matrix &m, std::string filename)
{
	cv::Mat m2 = matrix2cv(m,true);
	//cv::resize(m2, m2, cv::Size(0, 0), 4, 4);
	cv::imwrite(filename, m2);
}

void show(const Simple_cnn::matrix &m, float zoom = 1.0f, const char *win_name = "", int wait_ms=1)
{
	if (m.cols <= 0 || m.rows <= 0 || m.chans <= 0) return;
	cv::Mat cv_m = matrix2cv(m,false);
	
	double min_, max_;
	cv_m = cv_m.reshape(1);
	cv::minMaxIdx(cv_m, &min_, &max_);
	cv_m = cv_m - min_;
	max_ = max_ - min_;
	cv_m /= max_;
	//	cv_m += 1.f;
	//	cv_m *= 0.5;
	cv_m = cv_m.reshape(m.chans, m.rows);

	if (zoom != 1.f) cv::resize(cv_m, cv_m, cv::Size(0, 0), zoom, zoom,0);
	cv::convertScaleAbs(cv_m, cv_m, 255.0);
	cv::imshow(win_name, cv_m);
	cv::waitKey(wait_ms);
}

// null name hides all windows	
void hide(const char *win_name = "")
{
	if (win_name == NULL) cv::destroyAllWindows();
	else cv::destroyWindow(win_name);
}

enum Simple_cnn_palette{ gray=0, hot=1, tensorglow=2, voodoo=3, saltnpepa=4};


cv::Mat colorize(cv::Mat im, Simple_cnn::Simple_cnn_palette color_palette = Simple_cnn::gray)
{

	if (im.cols <= 0 || im.rows <= 0) return im;

	cv::Mat RGB[3];
	RGB[0] = im.clone(); // blue
	RGB[1] = im.clone();
	RGB[2] = im.clone();

	for (int i = 0; i < im.rows*im.cols; i++)
	{
		unsigned char c = (unsigned char)im.data[i];
		// tensor flow colors (red black blue)
		if (color_palette == Simple_cnn::tensorglow)
		{
			if (c == 255) { RGB[2].data[i] = 255; RGB[1].data[i] = 255;  RGB[0].data[i] = 255; }
			else if (c < 128) { RGB[2].data[i] = 0; RGB[1].data[i] = 0; RGB[0].data[i] = 2*(127 - c); }
			else { RGB[2].data[i] = 2* (c - 128); RGB[1].data[i] = 0; RGB[0].data[i] = 0; }
		}
		else if (color_palette == Simple_cnn::hot)
		{
			if (c == 255) { RGB[2].data[i] = 255; RGB[1].data[i] = 255;  RGB[0].data[i] = 255; }
			else if (c < 128) { RGB[0].data[i] = 0; RGB[1].data[i] = 0; RGB[2].data[i] = c * 2; }
			else { RGB[0].data[i] = 0; RGB[1].data[i] = (c - 128) * 2; RGB[2].data[i] = 255; }
		}
		else if (color_palette == Simple_cnn::saltnpepa)
		{
			if (c == 255) { RGB[2].data[i] = 255; RGB[1].data[i] = 255;  RGB[0].data[i] = 255; }
			else if (c&1) { RGB[0].data[i] = 0; RGB[1].data[i] = 0; RGB[2].data[i] = 0; }
			else { RGB[0].data[i] = 255; RGB[1].data[i] = 255; RGB[2].data[i] = 255; }
		}
		else if (color_palette == Simple_cnn::voodoo)
		{
			if (c == 255) { RGB[2].data[i] = 255; RGB[1].data[i] = 255;  RGB[0].data[i] = 255; }
			else if (c < 128) 
			{ RGB[2].data[i] = (127-c); RGB[1].data[i] = 0; RGB[0].data[i] = 2 * (127 - c); }
			else { RGB[2].data[i] =  (c - 128); RGB[1].data[i] = 2*(c-128); RGB[0].data[i] = 0; }
		}
	}

	cv::Mat out;
	cv::merge(RGB, 3, out);
	return out;
	//cv::applyColorMap(im, im, cv::COLORMAP_WINTER);// COLORMAP_HOT); // cv::COLORMAP_JET); COLORMAP_RAINBOW
}

Simple_cnn::matrix draw_cnn_weights(Simple_cnn::network &cnn, int layer_index, Simple_cnn::Simple_cnn_palette color_palette= Simple_cnn::gray)
{
	int w = (int)cnn.W.size();
	cv::Mat im;


	std::vector <cv::Mat> im_layers;

	int layers = (int)cnn.layer_sets.size();
	//for (int k = 0; k < layers; k++)
	int k=layer_index-1;
	{
		base_layer *layer = cnn.layer_sets[k];
	//	if (dynamic_cast<convolution_layer*> (layer) == NULL)  return Simple_cnn::matrix();// continue;

		for(auto &link : layer->forward_linked_layers)
		{
			int connection_index = link.first;
			base_layer *p_bottom = link.second;
			if (!p_bottom->has_weights()) continue;
			for (auto i = 0; i < cnn.W[connection_index]->chans; i++)
			{
				cv::Mat im = matrix2cv(cnn.W[connection_index]->get_chans(i), true);
				cv::resize(im, im, cv::Size(0, 0), 2., 2., 0);
				im_layers.push_back(im);
			}
			// draw these nicely
			int s = im_layers[0].cols;
			cv::Mat tmp(layer->node.chans*(s + 1) + 1, p_bottom->node.chans*(s+1)+1, CV_8UC1);// = im.clone();
			tmp = 255;
			for (int j = 0; j < layer->node.chans; j++)
			{
				for (int i = 0; i < p_bottom->node.chans; i++)
				{
					// make colors go 0 to 254
					double min, max;
					int index = i+j*p_bottom->node.chans;
					cv::minMaxIdx(im_layers[index], &min, &max);
					im_layers[index] -= min;
					im_layers[index] /= (max - min) / 254;

					im_layers[index].convertTo(im_layers[index], CV_8UC1);
					im_layers[index].copyTo(tmp(cv::Rect(i*s + 1 + i, j*s+  1+j, s, s)));
				}
			}
			im = tmp;
		}
	}

	return cv2matrix(colorize(im, color_palette));
}

Simple_cnn::matrix draw_cnn_weights(Simple_cnn::network &cnn, std::string layer_name, Simple_cnn::Simple_cnn_palette color_palette = Simple_cnn::gray)
{
	int layer_index = cnn.layer_map[layer_name];
	return draw_cnn_weights(cnn, layer_index, color_palette);
}


Simple_cnn::matrix draw_cnn_state(Simple_cnn::network &cnn, int layer_index, Simple_cnn::Simple_cnn_palette color_palette = Simple_cnn::gray)
{
	cv::Mat im;
	int layers = (int)cnn.layer_sets.size();
	if (layer_index < 0 || layer_index >= layers) return Simple_cnn::matrix();

	std::vector <cv::Mat> im_layers;
	base_layer *layer = cnn.layer_sets[layer_index];

	for (int i = 0; i < layer->node.chans; i++)
	{
		cv::Mat im = matrix2cv(layer->node.get_chans(i), true);
		cv::resize(im, im, cv::Size(0, 0), 2., 2., 0);
		im_layers.push_back(im);
	}
	// draw these nicely
	int s = im_layers[0].cols;
	cv::Mat tmp(s + 2, (int)im_layers.size()*(1+s) + 1, CV_8UC1);// = im.clone();
	tmp = 255;
	for (int i = 0; i < im_layers.size(); i++)
	{
		// make colors go 0 to 254
		double min, max;
		cv::minMaxIdx(im_layers[i], &min, &max);
		im_layers[i] -= min;
		im_layers[i] /= (max - min) / 254;

		im_layers[i].convertTo(im_layers[i], CV_8UC1);
		im_layers[i].copyTo(tmp(cv::Rect(i*s + 1 + i, 1, s, s)));
	}
	im = tmp;

	return cv2matrix(colorize(im, color_palette));
}

Simple_cnn::matrix draw_cnn_state(Simple_cnn::network &cnn, std::string layer_name, Simple_cnn::Simple_cnn_palette color_palette = Simple_cnn::gray)
{
	int layer_index = cnn.layer_map[layer_name];
	return draw_cnn_state(cnn, layer_index, color_palette);
}



#endif // CNN_CV#

}// namespace