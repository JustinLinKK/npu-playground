#pragma once
#include <cstddef>
#include <cstdint>

namespace aie {
template <typename T, unsigned N> struct vector {
    T lanes[N];
    T &operator[](unsigned i) { return lanes[i]; }
    const T &operator[](unsigned i) const { return lanes[i]; }
};
template <unsigned N, typename T> vector<T, N> load_v(const T *p) {
    vector<T, N> v;
    for (unsigned i = 0; i < N; ++i) v[i] = p[i];
    return v;
}
template <typename T, unsigned N> void store_v(T *p, const vector<T, N> &v) {
    for (unsigned i = 0; i < N; ++i) p[i] = v[i];
}
template <typename T, unsigned N> vector<T, N> add(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> v;
    for (unsigned i = 0; i < N; ++i) v[i] = a[i] + b[i];
    return v;
}
template <typename T, unsigned N> T reduce_add(const vector<T, N> &a) {
    T sum = 0;
    for (unsigned i = 0; i < N; ++i) sum += a[i];
    return sum;
}
}
