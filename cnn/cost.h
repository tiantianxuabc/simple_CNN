
#pragma once

#include <math.h>
#include <algorithm>
#include <string>

namespace Simple_cnn {

	namespace mse
	{
		inline float  cost(float out, float target)
		{
			return 0.5f*(out - target)*(out - target);
		};
		inline float  d_cost(float out, float target)
		{
			return (out - target);
		};
		const char name[] = "mse";
	}

	namespace cross_entropy
	{
		inline float  cost(float out, float target)
		{
			return (-target * std::log(out) - (1.f - target) * std::log(1.f - out));
		};
		inline float  d_cost(float out, float target)
		{
			return ((out - target) / (out*(1.f - out)));
		};
		const char name[] = "cross_entropy";
	}


	typedef struct
	{
	public:
		float(*cost)(float, float);
		float(*d_cost)(float, float);
		const char *name;
	} cost_function;

	cost_function* new_cost_function(std::string loss)
	{
		cost_function *p = new cost_function;
		if (loss.compare(cross_entropy::name) == 0)
		{
			p->cost = &cross_entropy::cost;
			p->d_cost = &cross_entropy::d_cost;
			p->name = cross_entropy::name;
			return p;
		}
		if (loss.compare(mse::name) == 0)
		{
			p->cost = &mse::cost;
			p->d_cost = &mse::d_cost;
			p->name = mse::name;
			return p;
		}
		else
			delete p;
		return NULL;
	}

	cost_function* new_cost_function(const char *type)
	{
		std::string loss(type);
		return new_cost_function(loss);
	}

}