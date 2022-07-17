#ifndef _DATA_HPP_INCLUDED_
#define _DATA_HPP_INCLUDED_
#pragma once


#include <iostream>
#include <type_traits>
#include <xutility>




template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class Data
{
protected:
	using type = T;
	using difference_type = std::ptrdiff_t;
private:
	T data;
public:
	static_assert(!std::is_const_v<T>, "Cannot initialize a const template argument! Class uses Copy and Move semantic though");

	Data() = delete;
	Data(const type& rhs);

	Data(const Data& Rhs) = default;
	Data(Data&& Rhs) noexcept = default;

	Data& operator=(const Data& Rhs) = default;
	Data& operator=(Data&& Rhs) noexcept = default;

	T Get_Data() const noexcept;

	~Data() = default;
};

template <typename T, typename T0>
Data<T, T0>::Data(const type& rhs) :
	data(rhs)
{

	std::cout << "Initialized" << __FUNCTION__ << '\n';
}

template <typename T, typename T0>
T Data<T, T0>::Get_Data() const noexcept
{
	return data;
}

#endif /* _DATA_HPP_INCLUDED_ */