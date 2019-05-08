#pragma once

#include <math.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <random>
#include <algorithm> 
#include <immintrin.h>


namespace Simple_cnn
{

	enum pad_type { zero = 0, edge = 1, median_edge = 2 };


	inline float dot(const float *x1, const float *x2, const int size)
	{
		float v = 0;
		for (int i = 0; i < size; i++)
		{
			v += x1[i] * x2[i];
		}
		return v;
	}

	inline float unwrap_2d_dot(const float *top_out, const float *delta, const int _rows, int t_cols, int d_cols)
	{
		float v = 0;

		for (int j = 0; j < _rows; j++)
		{
			v += dot(&top_out[t_cols*j], &delta[d_cols*j], d_cols);
		}
		return v;
	}


	// second item is rotated 180 (this is a convolution)
	inline float dot_rot180(const float *x1, const float *x2, const int size)
	{
		float v = 0;
		for (int i = 0; i < size; i++)
		{
			v += x1[i] * x2[size - i - 1];
		}
		return v;

	}
	inline float unwrap_2d_dot_rot180(const float *x1, const float *x2, const int size, int stride1, int stride2)
	{
		float v = 0;
		for (int j = 0; j < size; j++)
		{
			v += dot_rot180(&x1[stride1*j], &x2[stride2*(size - j - 1)], size);
		}
		return v;
	}





	class matrix
	{
		int _size;
		int _capacity;

		void delete_x()
		{
			delete[] x;
			x = NULL;
		}
		float *new_x(const int size)
		{
			x = new float[size];
			return x;
		}
	public:
		std::string _name;
		int cols, rows, chans;
		int chan_stride;
		int chan_aligned;
		float *x;
		// size must be divisible by 8 for AVX
		virtual int calc_chan_stride(int w, int h)
		{
			if (chan_aligned)
			{
				int s = w * h;
				const int remainder = s % 8;
				if (remainder > 0)
					s += 8 - remainder;
				return s;
			}
			else return w * h;
		}

		matrix() : cols(0), rows(0), chans(0), _size(0), _capacity(0), chan_stride(0), x(NULL), chan_aligned(0) {}


		matrix(int _w, int _h, int _c = 1, const float *data = NULL, int align_chan = 0) : cols(_w), rows(_h), chans(_c)
		{
			chan_aligned = align_chan;
			chan_stride = calc_chan_stride(cols, rows);
			_size = chan_stride * chans; _capacity = _size; x = new_x(_size);
			if (data != NULL)
				memcpy(x, data, _size * sizeof(float));
		}

		// copy constructor - deep copy
		matrix(const matrix &m) : cols(m.cols), rows(m.rows), chan_aligned(m.chan_aligned), chans(m.chans), chan_stride(m.chan_stride), _size(m._size), _capacity(m._size)
		{
			x = new_x(_size); memcpy(x, m.x, sizeof(float)*_size);
		}
		// copy and pad constructor
		matrix(const matrix &m, int pad_cols, int pad_rows, Simple_cnn::pad_type padding = Simple_cnn::zero) : cols(m.cols), rows(m.rows), chans(m.chans), chan_aligned(m.chan_aligned), chan_stride(m.chan_stride), _size(m._size), _capacity(m._size)
		{
			x = new_x(_size);
			memcpy(x, m.x, sizeof(float)*_size);
			*this = pad(pad_cols, pad_rows, padding);
		}

		~matrix()
		{
			if (x)
				delete_x();
		}

		matrix get_chans(int start_channel, int num_chans = 1) const
		{
			return matrix(cols, rows, num_chans, &x[start_channel*chan_stride]);
		}


		// if edge_pad==0, then the padded area is just 0. 
		// if edge_pad==1 it fills with edge pixel colors
		// if edge_pad==2 it fills with median edge pixel color
		matrix pad(int dx, int dy, Simple_cnn::pad_type edge_pad = Simple_cnn::zero) const
		{
			return pad(dx, dy, dx, dy, edge_pad);
		}
		matrix pad(int dx, int dy, int dx_right, int dy_bottom, Simple_cnn::pad_type edge_pad = Simple_cnn::zero) const
		{
			matrix v(cols + dx + dx_right, rows + dy + dy_bottom, chans);
			v.fill(0);

			//float *new_x = new float[chans*w*h]; 
			for (int k = 0; k < chans; k++)
			{
				const int v_chan_offset = k * v.chan_stride;
				const int chan_offset = k * chan_stride;
				// find median color of perimeter
				float median = 0.f;
				if (edge_pad == Simple_cnn::median_edge)
				{
					int perimeter = 2 * (cols + rows - 2);
					std::vector<float> d(perimeter);
					for (int i = 0; i < cols; i++)
					{
						d[i] = x[i + chan_offset]; d[i + cols] = x[i + cols * (rows - 1) + chan_offset];
					}
					for (int i = 1; i < (rows - 1); i++)
					{
						d[i + cols * 2] = x[cols*i + chan_offset];
						// file from back so i dont need to cal index
						d[perimeter - i] = x[cols - 1 + cols * i + chan_offset];
					}

					std::nth_element(d.begin(), d.begin() + perimeter / 2, d.end());
					median = d[perimeter / 2];
				}

				for (int j = 0; j < rows; j++)
				{
					memcpy(&v.x[dx + (j + dy)*v.cols + v_chan_offset], &x[j*cols + chan_offset], sizeof(float)*cols);
					if (edge_pad == Simple_cnn::edge)
					{
						// do left/right side
						for (int i = 0; i < dx; i++) v.x[i + (j + dy)*v.cols + v_chan_offset] = x[0 + j * cols + chan_offset];
						for (int i = 0; i < dx_right; i++) v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = x[(cols - 1) + j * cols + chan_offset];
					}
					else if (edge_pad == Simple_cnn::median_edge)
					{
						for (int i = 0; i < dx; i++) v.x[i + (j + dy)*v.cols + v_chan_offset] = median;
						for (int i = 0; i < dx_right; i++) v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = median;
					}
				}
				// top bottom pad
				if (edge_pad == Simple_cnn::edge)
				{
					for (int j = 0; j < dy; j++)	memcpy(&v.x[(j)*v.cols + v_chan_offset], &v.x[(dy)*v.cols + v_chan_offset], sizeof(float)*v.cols);
					for (int j = 0; j < dy_bottom; j++) memcpy(&v.x[(j + dy + rows)*v.cols + v_chan_offset], &v.x[(rows - 1 + dy)*v.cols + v_chan_offset], sizeof(float)*v.cols);
				}
				if (edge_pad == Simple_cnn::median_edge)
				{
					for (int j = 0; j < dy; j++)
						for (int i = 0; i < v.cols; i++)
							v.x[i + j * v.cols + v_chan_offset] = median;
					for (int j = 0; j < dy_bottom; j++)
						for (int i = 0; i < v.cols; i++)
							v.x[i + (j + dy + rows)*v.cols + v_chan_offset] = median;
				}
			}

			return v;
		}

		matrix crop(int dx, int dy, int w, int h) const
		{
			matrix v(w, h, chans);


			for (int k = 0; k < chans; k++)
			{
				for (int j = 0; j < h; j++)
				{
					memcpy(&v.x[j*w + k * v.chan_stride], &x[dx + (j + dy)*cols + k * chan_stride], sizeof(float)*w);
				}
			}

			return v;
		}

		Simple_cnn::matrix shift(int dx, int dy, Simple_cnn::pad_type edge_pad = Simple_cnn::zero)
		{
			int orig_cols = cols;
			int orig_rows = rows;
			int off_x = abs(dx);
			int off_y = abs(dy);

			Simple_cnn::matrix shifted = pad(off_x, off_y, edge_pad);

			return shifted.crop(off_x - dx, off_y - dy, orig_cols, orig_rows);
		}

		Simple_cnn::matrix flip_cols()
		{
			Simple_cnn::matrix v(cols, rows, chans);
			for (int k = 0; k < chans; k++)
				for (int j = 0; j < rows; j++)
					for (int i = 0; i < cols; i++)
						v.x[i + j * cols + k * chan_stride] = x[(cols - i - 1) + j * cols + k * chan_stride];

			return v;
		}
		Simple_cnn::matrix flip_rows()
		{
			Simple_cnn::matrix v(cols, rows, chans);

			for (int k = 0; k < chans; k++)
				for (int j = 0; j < rows; j++)
					memcpy(&v.x[(rows - 1 - j)*cols + k * chan_stride], &x[j*cols + k * chan_stride], cols * sizeof(float));

			return v;
		}

		void clip(float min, float max)
		{
			int s = chan_stride * chans;
			for (int i = 0; i < s; i++)
			{
				if (x[i] < min) x[i] = min;
				if (x[i] > max) x[i] = max;
			}
		}


		void min_max(float *min, float *max, int *min_i = NULL, int *max_i = NULL)
		{
			int s = rows * cols;
			int mini = 0;
			int maxi = 0;
			for (int c = 0; c < chans; c++)
			{
				const int t = chan_stride * c;
				for (int i = t; i < t + s; i++)
				{
					if (x[i] < x[mini]) mini = i;
					if (x[i] > x[maxi]) maxi = i;
				}
			}
			*min = x[mini];
			*max = x[maxi];
			if (min_i) *min_i = mini;
			if (max_i) *max_i = maxi;
		}

		float mean()
		{
			const int s = rows * cols;
			int cnt = 0;// channel*s;
			float average = 0;
			for (int c = 0; c < chans; c++)
			{
				const int t = chan_stride * c;
				for (int i = 0; i < s; i++)
					average += x[i + t];
			}
			average = average / (float)(s*chans);
			return average;
		}
		float remove_mean(int channel)
		{
			int s = rows * cols;
			int offset = channel * chan_stride;
			float average = 0;
			for (int i = 0; i < s; i++) average += x[i + offset];
			average = average / (float)s;
			for (int i = 0; i < s; i++) x[i + offset] -= average;
			return average;
		}

		float remove_mean()
		{
			float m = mean();
			int s = chan_stride * chans;
			//int offset = channel*s;
			for (int i = 0; i < s; i++)
				x[i] -= m;
			return m;
		}
		void fill(float val)
		{
			for (int i = 0; i < _size; i++)
				x[i] = val;
		}
		void fill_random_uniform(float range)
		{
			std::mt19937 gen(0);
			std::uniform_real_distribution<float> dst(-range, range);
			for (int i = 0; i < _size; i++)
				x[i] = dst(gen);
		}
		void fill_random_normal(float std)
		{
			std::mt19937 gen(0);
			std::normal_distribution<float> dst(0, std);
			for (int i = 0; i < _size; i++)
				x[i] = dst(gen);
		}


		// deep copy
		inline matrix& operator =(const matrix &m)
		{
			resize(m.cols, m.rows, m.chans, m.chan_aligned);
			memcpy(x, m.x, sizeof(float)*_size);
			return *this;
		}

		int  size() const
		{
			return _size;
		}

		void resize(int _w, int _h, int _c, int align_chans = 0)
		{
			chan_aligned = align_chans;
			int new_stride = calc_chan_stride(_w, _h);
			int s = new_stride * _c;
			if (s > _capacity)
			{
				if (_capacity > 0) delete_x(); _size = s; _capacity = _size; x = new_x(_size);
			}
			cols = _w; rows = _h; chans = _c; _size = s; chan_stride = new_stride;
		}

		// dot vector to 2d mat
		inline matrix dot_1dx2d(const matrix &m_2d) const
		{
			Simple_cnn::matrix v(m_2d.rows, 1, 1);
			for (int j = 0; j < m_2d.rows; j++)	v.x[j] = dot(x, &m_2d.x[j*m_2d.cols], _size);
			return v;
		}

		// +=
		inline matrix& operator+=(const matrix &m2)
		{
			for (int i = 0; i < _size; i++)
				x[i] += m2.x[i];
			return *this;
		}
		// -=
		inline matrix& operator-=(const matrix &m2)
		{
			for (int i = 0; i < _size; i++)
				x[i] -= m2.x[i];
			return *this;
		}

		// *= float
		inline matrix operator *=(const float v)
		{
			for (int i = 0; i < _size; i++)
				x[i] = x[i] * v;
			return *this;
		}


		// *= matrix
		inline matrix operator *=(const matrix &v)
		{
			for (int i = 0; i < _size; i++) x[i] = x[i] * v.x[i];
			return *this;
		}
		inline matrix operator *(const matrix &v)
		{
			matrix T(cols, rows, chans);
			for (int i = 0; i < _size; i++) T.x[i] = x[i] * v.x[i];
			return T;
		}
		// * float
		inline matrix operator *(const float v)
		{
			matrix T(cols, rows, chans);
			for (int i = 0; i < _size; i++)
				T.x[i] = x[i] * v;
			return T;
		}

		// + float
		inline matrix operator +(const float v)
		{
			matrix T(cols, rows, chans);
			for (int i = 0; i < _size; i++)
				T.x[i] = x[i] + v;
			return T;
		}

		// +
		inline matrix operator +(matrix m2)
		{
			matrix T(cols, rows, chans);
			for (int i = 0; i < _size; i++)
				T.x[i] = x[i] + m2.x[i];
			return T;
		}
	};
}

