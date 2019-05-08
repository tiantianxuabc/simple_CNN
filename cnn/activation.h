#pragma once

#include <math.h>
#include <algorithm>
#include <string>

namespace Simple_cnn
{


	// not using class because I thought this may be faster than vptrs
	namespace tan_h
	{
		inline void f(float *in, const int size, const float *bias) // this is activation f(x)
		{
			for (int i = 0; i < size; i++)
			{
				const float ep = std::exp((in[i] + bias[i]));
				const float em = std::exp(-(in[i] + bias[i]));
				in[i] = (ep - em) / (ep + em);
			}
		}
		inline void fc(float *in, const int size, const float bias) // this is activation f(x)
		{
			for (int i = 0; i < size; i++)
			{
				const float ep = std::exp((in[i] + bias));
				const float em = std::exp(-(in[i] + bias));
				in[i] = (ep - em) / (ep + em);
			}
		}
		inline float  df(float *in, int i, const int size)
		{
			return (1.f - in[i] * in[i]);
		}  // this is df(x), but we pass in the activated value f(x) and not x 
		const char name[] = "tanh";
	}

	namespace elu
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			//std::cout << " in + bias" << std::endl;
			for (int i = 0; i < size; i++)
			{
				if ((in[i] + bias[i]) < 0)
					in[i] = 0.1f*(std::exp((in[i] + bias[i])) - 1.f);
				else
					in[i] = (in[i] + bias[i]);
			}

		}
		inline void  fc(float *in, const int size, const float bias)
		{
			//std::cout << " in + bias" << std::endl;
			for (int i = 0; i < size; i++)
			{
				if ((in[i] + bias) < 0)
					in[i] = 0.1f*(std::exp((in[i] + bias)) - 1.f);
				else
					in[i] = (in[i] + bias);
			}

		}
		inline float  df(float *in, int i, const int size)
		{
			if (in[i] > 0)
				return 1.f;
			else
				return 0.1f*std::exp(in[i]);
		}
		const char name[] = "elu";
	}

	namespace identity
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			for (int i = 0; i < size; i++) 
				in[i] = (in[i] + bias[i]);
		}
		inline void  fc(float *in, const int size, const float bias)
		{
			for (int i = 0; i < size; i++) 
				in[i] = (in[i] + bias);
		}
		inline float  df(float *in, int i, const int size)
		{
			return 1.f;
		};
		const char name[] = "identity";
	}

	namespace relu
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			for (int i = 0; i < size; i++)
			{
				if ((in[i] + bias[i]) < 0) in[i] = 0;
				else in[i] = (in[i] + bias[i]);
			}
		}
		inline void  fc(float *in, const int size, const float bias)
		{
			for (int i = 0; i < size; i++)
			{
				if ((in[i] + bias) < 0) in[i] = 0;
				else in[i] = (in[i] + bias);
			}
		}
		inline float  df(float *in, int i, const int size)
		{
			if (in[i] > 0)
				return 1.0f;
			else
				return 0.0f;
		}
		const char name[] = "relu";
	};

	namespace lrelu
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			for (int i = 0; i < size; i++) {
				if ((in[i] + bias[i]) < 0) in[i] = 0.01f*(in[i] + bias[i]);
				else in[i] = (in[i] + bias[i]);
			}
		}
		inline void  fc(float *in, const int size, const float bias)
		{
			for (int i = 0; i < size; i++) {
				if ((in[i] + bias) < 0) in[i] = 0.01f*(in[i] + bias);
				else in[i] = (in[i] + bias);
			}
		}
		inline float  df(float *in, int i, const int size)
		{
			if (in[i] > 0)
				return 1.0f;
			else
				return 0.01f;
		}
		const char name[] = "lrelu";
	};

	namespace vlrelu
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			for (int i = 0; i < size; i++) {
				if ((in[i] + bias[i]) < 0) in[i] = 0.33f*(in[i] + bias[i]);
				else in[i] = (in[i] + bias[i]);
			}
		}
		inline void  fc(float *in, const int size, const float bias)
		{
			for (int i = 0; i < size; i++) {
				if ((in[i] + bias) < 0) in[i] = 0.33f*(in[i] + bias);
				else in[i] = (in[i] + bias);
			}
		}
		inline float  df(float *in, int i, const int size) { if (in[i] > 0) return 1.0f; else return 0.33f; }
		const char name[] = "vlrelu";
	};

	namespace sigmoid
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			for (int i = 0; i < size; i++)  in[i] = 1.0f / (1.0f + exp(-(in[i] + bias[i])));
		}
		inline void  fc(float *in, const int size, const float bias)
		{
			for (int i = 0; i < size; i++)  in[i] = 1.0f / (1.0f + exp(-(in[i] + bias)));
		}
		inline float df(float *in, int i, const int size) { return in[i] * (1.f - in[i]); }
		const char name[] = "sigmoid";
	};


	namespace softmax
	{
		inline void f(float *in, const int size, const float *bias)
		{
			float max = in[0];
			for (int j = 1; j < size; j++)
				if (in[j] > max) max = in[j];

			float denom = 0;
			for (int j = 0; j < size; j++) denom += std::exp(in[j] - max);

			for (int i = 0; i < size; i++) in[i] = std::exp(in[i] - max) / denom;
		}
		inline void fc(float *in, const int size, const float bias)
		{
			float max = in[0];
			for (int j = 1; j < size; j++) if (in[j] > max) max = in[j];

			float denom = 0;
			for (int j = 0; j < size; j++) denom += std::exp(in[j] - max);

			for (int i = 0; i < size; i++) in[i] = std::exp(in[i] - max) / denom;
		}
		inline float df(float *in, int i, const int size)
		{
			// don't really use... should use good cost func to make this go away
			return in[i] * (1.f - in[i]);		
		}

		const char name[] = "softmax";
	};


	namespace brokemax
	{
		inline void f(float *in, const int size, const float *bias)
		{
			for (int i = 0; i < size; i++)
			{
				float max = in[0];
				for (int j = 1; j < size; j++) if (in[j] > max) max = in[j];

				float denom = 0;
				for (int j = 0; j < size; j++) denom += std::exp(in[j] - max);

				in[i] = std::exp(in[i] - max) / denom;
			}

		}
		inline void fc(float *in, const int size, const float bias)
		{
			float max = in[0];
			for (int j = 1; j < size; j++) if (in[j] > max) max = in[j];

			float denom = 0;
			for (int j = 0; j < size; j++) denom += std::exp(in[j] - max);

			for (int i = 0; i < size; i++) in[i] = std::exp(in[i] - max) / denom;
		}
		inline float df(float *in, int i, const int size)
		{
			// don't really use... should use good cost func to make this go away
			return in[i] * (1.f - in[i]);
			//		for(int j=0; j<size; j++) 
			//		{
			//			if(i==j) in[i]= in[i] * (1.f - in[i]);
			//			else in[i] = in[i]*in[j];
			//		}
		}

		const char name[] = "brokemax";
	};

	namespace none
	{
		inline void f(float *in, const int size, const float *bias) { return; };
		inline void fc(float *in, const int size, const float bias) { return; };
		inline float df(float *in, int i, int size) { return 0; };
		const char name[] = "none";
	};

	typedef struct
	{
	public:
		void(*f)(float *, const int, const float*);
		void(*fc)(float *, const int, const float);
		float(*df)(float *, int, const int);
		const char *name;
	} activation_function;

	activation_function* new_activation_function(std::string act)
	{
		activation_function *p = new activation_function;
		if (act.compare(tan_h::name) == 0) { p->f = &tan_h::f; p->fc = &tan_h::fc; p->df = &tan_h::df; p->name = tan_h::name; return p; }
		if (act.compare(identity::name) == 0) { p->f = &identity::f; p->fc = &identity::fc; p->df = &identity::df; p->name = identity::name; return p; }
		if (act.compare(vlrelu::name) == 0) { p->f = &vlrelu::f; p->fc = &vlrelu::fc; p->df = &vlrelu::df; p->name = vlrelu::name; return p; }
		if (act.compare(lrelu::name) == 0) { p->f = &lrelu::f; p->fc = &lrelu::fc; p->df = &lrelu::df; p->name = lrelu::name; return p; }
		if (act.compare(relu::name) == 0) { p->f = &relu::f; p->fc = &relu::fc; p->df = &relu::df; p->name = relu::name; return p; }
		if (act.compare(sigmoid::name) == 0) { p->f = &sigmoid::f; p->fc = &sigmoid::fc; p->df = &sigmoid::df; p->name = sigmoid::name; return p; }
		if (act.compare(elu::name) == 0) { p->f = &elu::f; p->fc = &elu::fc; p->df = &elu::df; p->name = elu::name; return p; }
		if (act.compare(none::name) == 0) { p->f = &none::f; p->fc = &none::fc; p->df = &none::df; p->name = none::name; return p; }
		if (act.compare(softmax::name) == 0) { p->f = &softmax::f; p->fc = &softmax::fc; p->df = &softmax::df; p->name = softmax::name; return p; }
		if (act.compare(brokemax::name) == 0) { p->f = &brokemax::f; p->fc = &brokemax::fc; p->df = &brokemax::df; p->name = brokemax::name; return p; }
		delete p;
		return NULL;
	}

	activation_function* new_activation_function(const char *type)
	{
		std::string act(type);
		return new_activation_function(act);
	}

} // namespace