
#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <Eigen>
#include <iostream>
#include <stdio.h>

#include "common.hpp"

#define BLOCK_SIZE 1024

using namespace std;

static const char *_cudaGetErrorEnum(cublasStatus_t error);

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

static __global__ void pow_elements(float *a, size_t N, float exponent) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
        a[tid] = powf(a[tid], exponent);
}

static __global__ void nonneg_elements(float *a, size_t N) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        if (a[tid] < 0)
            a[tid] = 0.0f;
    }
}

static __global__ void nonneg_and_pow_elements(float *a, size_t N, float exponent) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        if (a[tid] > 0.0f)
            a[tid] = powf(a[tid], exponent);
        else
            a[tid] = 0.0f;
    }
}

static __global__ void hadamard_elements(float *out, float *a, float *b, size_t N) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        out[tid] = a[tid] * b[tid];
    }
}

static __global__ void zero_elements(float *a, size_t N) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        a[tid] = 0.0f;
    }
}

static __global__ void sub_and_exp_elements(float *a, size_t N, float sub) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
        a[tid] = expf(a[tid]-sub);
}

static __global__ void sub_and_exp_and_add_elements(float *a, size_t N, float sub, float add) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
        a[tid] = expf(a[tid]-sub)+add;
}

static __global__ void inv_nz_elements(float *a, size_t N) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        if (a[tid] != 0.0f)
            a[tid] = 1.0f/a[tid];
    }
}

static __global__ void learning_rate_elements(float *a, size_t N, size_t base, float t0, float *coeffs=NULL) {
    /* which element does this compute? */
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (coeffs) {
        if (tid < N) {
            a[tid] = coeffs[tid] * sqrtf((1.0f + t0) / (float(base+tid+1) + t0));
        }
        else if ( (N <= tid) && (tid < 2*N) ) {
            a[tid] = 1.0f - sqrtf((1.0f + t0) / (float(base+(tid-N)+1) + t0));
        }
    }
    else {
        if (tid < N) {
            a[tid] = sqrtf((1.0f + t0) / (float(base+tid+1) + t0));
        }
        else if ( (N <= tid) && (tid < 2*N) ) {
            a[tid] = 1.0f - sqrtf((1.0f + t0) / (float(base+(tid-N)+1) + t0));
        }
    }
}

class CuVec;

struct CuMat {
    cublasHandle_t _handle;

    Eigen::MatrixXf _tmp_storage;

    float *_d_mat;
    size_t _rows;
    size_t _cols;

    cublasStatus_t _stat;

    CuMat()
        : _handle(0), _rows(0), _cols(0), _d_mat(NULL) {
    }

    CuMat(cublasHandle_t handle)
        : _handle(handle), _rows(0), _cols(0), _d_mat(NULL) {
    }

    CuMat(cublasHandle_t handle, const size_t rows, const size_t cols)
        : _handle(handle), _rows(rows), _cols(cols), _d_mat(NULL) {
        alloc();
    }

    CuMat(const CuMat &rhs)
        : _rows(0), _cols(0), _d_mat(NULL) {
        *this = rhs;
    }

    ~CuMat() {
        dealloc();
    }

    void alloc() {
        if (_d_mat)
            dealloc();

        cudaError_t cudaStat = cudaMalloc ( (void**) &_d_mat, (_rows*_cols) * sizeof(float) );
        //cerr << "alloc() " << _rows << " x " << _cols << endl;
    }

    void dealloc() {
        if (_d_mat) {
            cudaFree(_d_mat);
            _d_mat = NULL;
        }

        //cerr << "dealloc()" << endl;
    }

    CuMat& operator=(const CuMat &rhs) {
        if (rhs._d_mat) {
            _handle = rhs._handle;

            if ( (_rows != rhs._rows) || (_cols != rhs._cols) ) {
                _rows = rhs._rows;
                _cols = rhs._cols;
                alloc();
            }

            _stat = cublasScopy(_handle, _rows*_cols, rhs._d_mat, 1,
                                _d_mat, 1);

            assert(_stat == CUBLAS_STATUS_SUCCESS);
        }

        return *this;
    }

    CuMat& set_zero() {
        assert(_d_mat);

        const size_t n = _rows*_cols;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length)
                d_tail = _d_mat + blocks*blocksize;
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        zero_elements<<<blocks, blocksize>>>(_d_mat, n);

        if (d_tail) {
            zero_elements<<<tail_length, 1>>>(d_tail, tail_length);
        }

        return *this;
    }

    CuMat& set_mTm(const CuMat &A, const CuMat &B, const float alpha=1.0f, const float beta=0.0f) {
        assert(A._rows == B._rows);

        if ( (_rows != A._cols) || (_cols != B._cols) ) {
            _rows = A._cols;
            _cols = B._cols;
            alloc();
        }

        _stat = cublasSgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, _rows, _cols, B._rows,
                            &alpha, A._d_mat, A._rows,
                            B._d_mat, B._rows,
                            &beta, _d_mat, _rows);


        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    CuMat& set_mmT(const CuMat &A, const CuMat &B, const float alpha=1.0f, const float beta=0.0f) {
        assert(A._cols == B._cols);

        if ( (_rows != A._rows) || (_cols != B._rows) ) {
            _rows = A._rows;
            _cols = B._rows;
            alloc();
            cerr << "mmT called resize!" << endl;
        }

        _stat = cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_T, _rows, _cols, A._cols,
                            &alpha, A._d_mat, A._rows,
                            B._d_mat, B._rows,
                            &beta, _d_mat, _rows);


        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    CuMat& set_mm(const CuMat &A, const CuMat &B, const float alpha=1.0f, const float beta=0.0f) {
        assert(A._cols == B._rows);

        if ( (_rows != A._rows) || (_cols != B._cols) ) {
            _rows = A._rows;
            _cols = B._cols;
            alloc();
            cerr << "mm called resize!" << endl;
        }

        _stat = cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _rows, _cols, A._cols,
                            &alpha, A._d_mat, A._rows,
                            B._d_mat, B._rows,
                            &beta, _d_mat, _rows);

        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    CuMat& set_dmm(const CuVec &dm, bool left=true);

    CuMat& set_add(const CuMat &A, const float alpha=1.0f, const float beta=1.0f) {
        assert(_d_mat);
        assert( (_rows == A._rows) && (_cols == A._cols) );

        //_stat = cublasSaxpy(_handle, _rows*_cols, &alpha, A._d_mat, 1, _d_mat, 1);
        _stat = cublasSgeam(_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            A._rows, _cols,
                            &alpha, A._d_mat, A._rows,
                            &beta, _d_mat, _rows,
                            _d_mat, _rows);

        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    CuMat& set_add(const CuMat &A, const CuMat &B, const float alpha=1.0f, const float beta=1.0f) {
        assert( (B._rows == A._rows) && (B._cols == A._cols) );

        if ( (_rows != A._rows) || (_cols != B._cols) ) {
            _rows = A._rows;
            _cols = A._cols;
            alloc();

            cerr << "alloc() called in set_add (2)" << endl;
        }

        //_stat = cublasSaxpy(_handle, _rows*_cols, &alpha, A._d_mat, 1, _d_mat, 1);
        _stat = cublasSgeam(_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            A._rows, _cols,
                            &alpha, A._d_mat, A._rows,
                            &beta, B._d_mat, B._rows,
                            _d_mat, _rows);

        cudaCheckErrors("set_add (2)");
        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    CuMat& nonneg_and_pow(const float exponent) {
        assert(_d_mat);

        const size_t n = _rows*_cols;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length)
                d_tail = _d_mat + blocks*blocksize;
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        // cerr << "n = " << n << endl
        //      << "blocksize = " << blocksize << endl
        //      << "blocks = " << blocks << endl
        //      << "d_tail = " << d_tail << endl
        //      << "tail_length = " << tail_length << endl
        //      << "exponent = " << exponent << endl;

        nonneg_and_pow_elements<<<blocks, blocksize>>>(_d_mat, n, exponent);
        cudaCheckErrors("nonneg_and_pow");

        if (d_tail) {
            nonneg_and_pow_elements<<<tail_length, 1>>>(d_tail, tail_length, exponent);
            cudaCheckErrors("nonneg_and_pow");
        }

        return *this;
    }

    CuMat& pow(const float exponent) {
        assert(_d_mat);

        const size_t n = _rows*_cols;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length)
                d_tail = _d_mat + blocks*blocksize;
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        pow_elements<<<blocks, blocksize>>>(_d_mat, n, exponent);
        cudaCheckErrors("pow");

        if (d_tail) {
            pow_elements<<<tail_length, 1>>>(d_tail, tail_length, exponent);
            cudaCheckErrors("pow");
        }

        return *this;
    }

    CuVec& get_col(CuVec &v, const size_t i) const;

    CuMat& rank1_upd(const CuVec &x, const CuVec &y, float alpha);

    CuMat& scale(const float alpha) {
        assert(_d_mat);

        _stat = cublasSscal(_handle, _rows*_cols, &alpha, _d_mat, 1);
        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    float trace_mTm(const CuMat &rhs);

    CuMat& outer(const CuVec &a, const CuVec &b);

    CuMat& learning_rates(const size_t base, const float t0, const size_t m, const CuVec *coeffs=NULL);

    CuMat& set_hadamard(const CuMat &rhs) {
        assert(_d_mat && rhs._d_mat);
        assert( (_rows == rhs._rows) && (_cols == rhs._cols) );

        const size_t n = _rows*_cols;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail_1 = NULL, *d_tail_2 = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length) {
                d_tail_1 = _d_mat + blocks*blocksize;
                d_tail_2 = rhs._d_mat + blocks*blocksize;
            }
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        hadamard_elements<<<blocks, blocksize>>>(_d_mat, _d_mat, rhs._d_mat, n);

        if (d_tail_1) {
            hadamard_elements<<<tail_length, 1>>>(_d_mat, d_tail_1, d_tail_2, tail_length);
        }

        return *this;
    }

    void from_eigen(const Eigen::MatrixXf &mat) {
        _rows = mat.rows();
        _cols = mat.cols();
        alloc();

        _stat = cublasSetMatrix(_rows, _cols, sizeof(float), mat.data(),
                                mat.rows(), _d_mat, _rows);

        // cerr << _cudaGetErrorEnum(_stat) << endl;
        // cudaCheckErrors("from_eigen");

        assert(_stat == CUBLAS_STATUS_SUCCESS);
    }

    void to_eigen(Eigen::MatrixXf &mat) {
        assert(_d_mat);

        if ( (mat.cols() != _cols) || (mat.rows() != _rows) )
            mat.resize(_rows, _cols);

        _stat = cublasGetMatrix(mat.rows(), mat.cols(), sizeof(float), _d_mat,
                                _rows, mat.data(), mat.rows());

        assert(_stat == CUBLAS_STATUS_SUCCESS);
    }

    Eigen::MatrixXf to_eigen() {
        Eigen::MatrixXf mat(_rows, _cols);
        to_eigen(mat);
        return mat;
    }

    void move_to_host() {
        to_eigen(_tmp_storage);
        dealloc();
    }

    void move_to_device() {
        from_eigen(_tmp_storage);
        _tmp_storage.resize(0,0);
    }

    ostream& save(ostream &out) {
        assert(_d_mat);

        Eigen::MatrixXf tmp_mat;
        to_eigen(tmp_mat);

        typename Eigen::MatrixXf::Index rows=tmp_mat.rows(), cols=tmp_mat.cols();
        out.write((char*) (&rows), sizeof(typename Eigen::MatrixXf::Index));
        out.write((char*) (&cols), sizeof(typename Eigen::MatrixXf::Index));
        out.write((char*) tmp_mat.data(), rows*cols*sizeof(typename Eigen::MatrixXf::Scalar) );

        return out;
    }

    istream& load(istream &in, const bool to_gpu=true) {
        typename Eigen::MatrixXf::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Eigen::MatrixXf::Index));
        in.read((char*) (&cols),sizeof(typename Eigen::MatrixXf::Index));

        _tmp_storage.resize(rows, cols);

        in.read( (char *) _tmp_storage.data() , rows*cols*sizeof(typename Eigen::MatrixXf::Scalar) );

        if (to_gpu) {
            from_eigen(_tmp_storage);
            _tmp_storage.resize(0,0);
        }

        return in;
    }
};

struct CuVec {
    cublasHandle_t _handle;

    float *_d_vec;
    size_t _size;
    float _slave;

    cublasStatus_t _stat;

    CuVec(cublasHandle_t handle)
        : _handle(handle), _size(0), _d_vec(NULL), _slave(false) {
    }

    CuVec(cublasHandle_t handle, const size_t size)
        : _handle(handle), _size(size), _d_vec(NULL), _slave(false) {
        alloc();
    }

    CuVec(const CuVec &rhs)
        : _size(0), _d_vec(NULL), _slave(false) {
        *this = rhs;
    }

    ~CuVec() {
        dealloc();
    }

    void alloc() {
        if (_d_vec)
            dealloc();

        cudaError_t cudaStat = cudaMalloc ( (void**) &_d_vec, _size * sizeof(float) );
    }

    void dealloc() {
        if (_slave)
            return;

        if (_d_vec) {
            cudaFree(_d_vec);
            _d_vec = NULL;
        }
    }

    CuVec& operator=(const CuVec &rhs) {
        if (rhs._d_vec) {
            _handle = rhs._handle;

            if (_size != rhs._size) {
                _size = rhs._size;
                alloc();
            }

            _stat = cublasScopy(_handle, _size, rhs._d_vec, 1,
                                _d_vec, 1);

            assert(_stat == CUBLAS_STATUS_SUCCESS);
        }

        return *this;
    }

    void from_eigen(const Eigen::VectorXf &vec) {
        if (_size != vec.size()) {
            _size = vec.size();
            alloc();
        }

        _stat = cublasSetVector(_size, sizeof(float), vec.data(), 1,
                                _d_vec, 1);

        assert(_stat == CUBLAS_STATUS_SUCCESS);
    }

    void to_eigen(Eigen::VectorXf &vec) {
        assert(_d_vec);

        vec.resize(_size);

        _stat = cublasGetVector(_size, sizeof(float), _d_vec, 1,
                                vec.data(), 1);

        // cerr << _cudaGetErrorEnum(_stat) << endl;
        // cudaCheckErrors("to_eigen");

        assert(_stat == CUBLAS_STATUS_SUCCESS);
    }

    Eigen::VectorXf to_eigen() {
        Eigen::VectorXf vec(_size);
        to_eigen(vec);
        return vec;
    }

    CuVec& set_mv(const CuMat &mat, const CuVec &vec, bool tranpose_mat=false, const float alpha=1.0f, const float beta=0.0f) {
        cublasOperation_t transa;

        if (!tranpose_mat) {
            if (_size != mat._rows) {
                _size = mat._rows;
                alloc();
            }

            transa = CUBLAS_OP_N;

            assert(mat._cols == vec._size);
        }
        else {
            if (_size != mat._cols) {
                _size = mat._cols;
                alloc();
            }

            transa = CUBLAS_OP_T;

            assert(mat._rows == vec._size);
        }

        _stat = cublasSgemv(_handle, transa, mat._rows, mat._cols, &alpha,
                            mat._d_mat, mat._rows, vec._d_vec, 1, &beta,
                            _d_vec, 1);

        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return *this;
    }

    CuVec& pow(const float exponent) {
        assert(_d_vec);

        const size_t n = _size;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length)
                d_tail = _d_vec + blocks*blocksize;
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        pow_elements<<<blocks, blocksize>>>(_d_vec, _size, exponent);

        if (d_tail) {
            pow_elements<<<tail_length, 1>>>(d_tail, tail_length, exponent);
        }

        return *this;
    }

    CuVec& sub_and_exp(const float sub) {
        assert(_d_vec);

        const size_t n = _size;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length)
                d_tail = _d_vec + blocks*blocksize;
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        sub_and_exp_elements<<<blocks, blocksize>>>(_d_vec, n, sub);

        if (d_tail) {
            sub_and_exp_elements<<<tail_length, 1>>>(d_tail, tail_length, sub);
        }

        return *this;
    }

    CuVec& inv_nz() {
        assert(_d_vec);

        const size_t n = _size;
        size_t blocksize = BLOCK_SIZE;
        size_t blocks = 0;
        float *d_tail = NULL;
        size_t tail_length = 0;

        if (n > blocksize) {
            blocks = n / blocksize;
            tail_length = n % blocksize;
            if (tail_length)
                d_tail = _d_vec + blocks*blocksize;
        }
        else {
            blocks = n;
            blocksize = 1;
        }

        inv_nz_elements<<<blocks, blocksize>>>(_d_vec, n);

        if (d_tail) {
            inv_nz_elements<<<tail_length, 1>>>(d_tail, tail_length);
        }

        return *this;
    }

    float amax() {
        assert(_d_vec);

        int max_val_ix;
        _stat = cublasIsamax(_handle, _size, _d_vec, 1, &max_val_ix);
        assert(_stat == CUBLAS_STATUS_SUCCESS);

        float max_val = 0.0f;
        _stat = cublasGetVector(1, sizeof(float), _d_vec+(max_val_ix-1), 1,
                                &max_val, 1);
        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return max_val;
    }

    float asum() {
        assert(_d_vec);

        float result;
        _stat = cublasSasum(_handle, _size, _d_vec, 1, &result);
        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return result;
    }

    float sum() {
        CuVec ones(_handle);
        ones.from_eigen(Eigen::VectorXf::Ones(_size));
        float out = 0.0f;

        _stat = cublasSdot(_handle, _size, _d_vec, 1, ones._d_vec, 1, &out);
        assert(_stat == CUBLAS_STATUS_SUCCESS);

        return out;
    }

};


inline CuMat& CuMat::set_dmm(const CuVec &dm, bool left) {
    if (!left)
        assert(dm._size == _cols);
    else
        assert(dm._size == _rows);


    assert(_d_mat);

    cublasSideMode_t mode = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    _stat = cublasSdgmm(_handle, mode,
                        _rows, _cols,
                        _d_mat, _rows,
                        dm._d_vec, 1,
                        _d_mat, _rows);

    assert(_stat == CUBLAS_STATUS_SUCCESS);

    return *this;
}

inline CuVec& CuMat::get_col(CuVec &v, const size_t i) const {
    assert(_d_mat);
    assert( i <= _cols );

    v._handle = _handle;
    v._d_vec = _d_mat + i*_rows;
    v._size = _rows;
    v._slave = true;

    return v;
}

inline CuMat& CuMat::rank1_upd(const CuVec &x, const CuVec &y, float alpha=1.0f) {
    assert(_d_mat);
    assert( (_rows == x._size) && (_cols == y._size) );

    _stat = cublasSger(_handle, _rows, _cols,
                       &alpha,
                       x._d_vec, 1,
                       y._d_vec, 1,
                       _d_mat, _rows);

    assert(_stat == CUBLAS_STATUS_SUCCESS);

    return *this;
}

inline CuMat& CuMat::learning_rates(const size_t base, const float t0, const size_t m, const CuVec *coeffs) {
    _rows = m;
    _cols = 2;
    alloc();

    float *d_coeffs = NULL;
    if (coeffs) {
        assert(coeffs->_size == m);
        d_coeffs = coeffs->_d_vec;
    }

    learning_rate_elements<<<_rows*_cols,1>>>(_d_mat, _rows, base, t0, d_coeffs);

    return *this;
}

inline float CuMat::trace_mTm(const CuMat &rhs) {
    assert(_d_mat && rhs._d_mat);
    assert( (_rows == rhs._rows) && (_cols == rhs._cols) );
    CuMat tmp(_handle, _rows, _cols);

    const size_t n = _rows*_cols;
    size_t blocksize = BLOCK_SIZE;
    size_t blocks = 0;
    float *d_tail_1 = NULL, *d_tail_2 = NULL;
    size_t tail_length = 0;

    if (n > blocksize) {
        blocks = n / blocksize;
        tail_length = n % blocksize;
        if (tail_length) {
            d_tail_1 = _d_mat + blocks*blocksize;
            d_tail_2 = rhs._d_mat + blocks*blocksize;
        }
    }
    else {
        blocks = n;
        blocksize = 1;
    }

    hadamard_elements<<<blocks, blocksize>>>(tmp._d_mat, _d_mat, rhs._d_mat, n);

    if (d_tail_1) {
        hadamard_elements<<<tail_length, 1>>>(tmp._d_mat, d_tail_1, d_tail_2, tail_length);
    }

    CuVec ones(_handle), sum_rows(_handle);
    ones.from_eigen(Eigen::VectorXf::Ones(tmp._cols));

    sum_rows.set_mv(tmp, ones, false);
    return sum_rows.sum();
}

inline CuMat& CuMat::outer(const CuVec &a, const CuVec &b) {
    if ( (_rows != a._size) || (_cols != b._size) ) {
        _rows = a._size;
        _cols = b._size;
        alloc();
    }

    set_zero();
    rank1_upd(a, b);
    return *this;
}

static cublasHandle_t init_cublas() {
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    cerr << _cudaGetErrorEnum(stat) << endl;
    return handle;
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static ostream& operator<<(ostream &os, CuMat &m) {
    os << m.to_eigen();
    return os;
}

static ostream& operator<<(ostream &os, CuVec &v) {
    os << v.to_eigen();
    return os;
}
