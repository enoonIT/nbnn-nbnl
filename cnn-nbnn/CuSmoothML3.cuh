
#pragma once

#include <fstream>
#include <stdint.h>
#include <vector>
#include <list>

#include "Timer.hpp"
#include "CuMat.cu"

#include "common.hpp"

#define __MAGIC__ "CUSMOOTHML3"
#define __MAGIC_LEN__ 11

typedef Eigen::MatrixXf Mat;
typedef Eigen::VectorXf Vec;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > MappedMat;

struct Minibatch {
    Mat *X;
    std::vector<uint32_t> *y;
    std::vector<uint32_t> *tags;
};

class MinibatchExampleSet {
public:
    virtual bool get_minibatch(Minibatch &e) = 0;
    virtual void reset() {}
};

struct CuHyperparameters {
    uint32_t n;
    float L;
    float L_start;
    float p;
    float lambda;
    float t0;
    float gauss_init_std;

    uint64_t pass_length;
};

class CuSmoothML3
{
public:
    std::vector<CuMat> _W;
    std::vector<CuMat> _W_bar;
    std::vector<CuMat> _G;

    unsigned long _t;
    float _sum_loss;
    CuHyperparameters _hp;
    float _q;
    float _W_norm_sum;

    std::string _model_dir;
    std::string _model_tag;
    size_t _save_every_t;

    bool _nbnl_mode;

    Timer _global_timer;

    float compute_class_scores(Mat &scores, vector<CuMat> &score_grads, const CuMat &X, const vector<uint32_t> &y, const bool dont_compute_grad=false) const;
    inline void update_G(CuMat &G, const CuMat &score_grads, const CuMat &cuX, const vector<uint32_t> &y,
                         const Vec &softmax_scores,
                         const float gamma_t, const size_t k);
    inline void update_G_convexified(CuMat &G, const CuMat &score_grads, const CuMat &pos_score_grads,
                                     const CuMat &cuX, const vector<uint32_t> &y,
                                     const Vec &softmax_scores,
                                     const float gamma_t, const size_t k);

    void adagrad_on_convexified(MinibatchExampleSet &example_set, const size_t classes, const size_t iter, float learning_rate=1.0f);
    void pegasos_on_convexified(MinibatchExampleSet &example_set, const size_t classes, const size_t iter);
    inline void compute_convexified_G(CuMat &G, const CuMat &cuX,
                                      const Vec &softmax_scores,
                                      const CuMat &score_grads,
                                      const CuMat &pos_score_grads,
                                      const vector<uint32_t> &y, const uint32_t k);


    inline void check_minibatch(const Minibatch &e);
    inline void try_init_models(const uint32_t classes, const uint32_t dim, const uint32_t n);
    void init_random_score_grads(vector<CuMat> &scores, const size_t classes, const size_t minibatch_size);

    float compute_surrogate(const float loss, vector<CuMat> &W_t_1, const vector<CuMat> &score_grads, const Mat &softmax_scores,
                            const CuMat &cuX, const vector<uint32_t> &y);

    void move_W_bar_to_dev();
    void move_W_bar_to_host();
    void move_G_to_dev();
    void move_G_to_host();

public:
    cublasHandle_t _handle;
    size_t _offload_num;

    double _last_train_duration_ms;
    double _last_test_duration_ms;

    bool _record_aer;
    std::list<float> _aer_track;

    MinibatchExampleSet *_validation_set;

    CuSmoothML3(const bool record_aer);

    void train(const uint32_t classes, const CuHyperparameters &hp, MinibatchExampleSet &example_set);
    void resume(const uint32_t classes, MinibatchExampleSet &example_set);
    size_t num_correct(const Minibatch &mb, const bool nbnl, uint32_t *nbnl_examples=0);
    size_t num_correct_cpu(const Mat &X, const vector<uint32_t> &y, const bool nbnl, uint32_t *nbnl_examples=0);
    float test(MinibatchExampleSet &example_set, const bool on_gpu=true);

    void set_storage_params(const std::string &path, const size_t save_every_t=500, const std::string model_tag="");
    void save();
    void load(const std::string &filename, const bool to_gpu=true);
};
