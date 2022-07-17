#ifndef _NEURON_HPP_INCLUDED_
#define _NEURON_HPP_INCLUDED_
#pragma once

#include <vector>
#include <memory>

#include "Data.hpp"

template<typename T>
class Neuron
{
protected:
	using type = Data<T>;
	using vector_type = std::unique_ptr<type>;
private:
	std::vector<type> input;
	std::vector<type> output;
public:
	Neuron() = default;

	Neuron(const std::initializer_list<type> Elems);

	~Neuron() = default;
};

template <typename T>
Neuron<T>::Neuron(const std::initializer_list<type> Elems) :
	input(Elems)
{ }

#endif /* _NEURON_HPP_INCLUDED_ */