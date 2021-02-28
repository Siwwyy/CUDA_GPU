#ifndef _NEURON_HPP_INCLUDED_
#define _NEURON_HPP_INCLUDED_
#pragma once

#include <vector>
#include <memory>

#include "Data.hpp"

template<typename T>
class Neuron
{
private:
	std::vector<std::unique_ptr<Data<T>>> input;
public:
	Neuron() = default;

	Neuron(const std::initializer_list<T>& Elems);

	~Neuron() = default;
};

template <typename T>
Neuron<T>::Neuron(const std::initializer_list<T>& Elems) :
	input(Elems)
{ }

#endif /* _NEURON_HPP_INCLUDED_ */
