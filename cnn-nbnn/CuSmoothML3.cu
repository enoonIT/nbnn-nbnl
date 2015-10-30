
#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <stdexcept>

#include "CuSmoothML3.cuh"

using std::setprecision;
using namespace std;
//using namespace std::chrono;

CuSmoothML3::CuSmoothML3(const bool record_aer) : _record_aer(record_aer), _offload_num(0), _save_every_t(0), _validation_set(0), _nbnl_mode(false) {
    _handle = init_cublas();
    srand((unsigned int) time(0));
    //cudaCheckErrors("init_cublas()");
}

void CuSmoothML3::train(const uint32_t classes, const CuHyperparameters &hp, MinibatchExampleSet &example_set) {
    _hp = hp;
    _t = 1;
    _sum_loss = 0.0f;
    _q = hp.p / (hp.p - 1.0f);
    _offload_num = classes/2;
    _W_norm_sum = 0.0f;
    _hp.L_start = _hp.L;

    // float W_bar_const = _hp.L / (_hp.L + hp.lambda);
    // float G_const = 1.0f / (_hp.L + hp.lambda);

    float W_bar_const = 1.0f / (1.0f + hp.lambda);
    float G_const = 1.0f / (_hp.L * (1.0f + hp.lambda) );

    Minibatch mb;
    CuMat score_grad(_handle, hp.n, classes);
    float log_sum_softmax;
    float loss;
    float gamma_t = 1.0f;

    if (hp.pass_length > 0) {
        pegasos_on_convexified(example_set, classes, hp.pass_length);
    }

    bool got_example = example_set.get_minibatch(mb);
    if (!got_example)
        return;

    const size_t minibatch_size = mb.X->cols();
    const size_t dim = mb.X->rows();

    __log__ << "[SMM] Starting training with " << classes << " classes." << endl
            << "[SMM] minibatch size = " << minibatch_size << endl
            << "[SMM] dimension = " << dim << endl
            << "[SMM] n = " << hp.n << endl
            << "[SMM] L = " << _hp.L << endl
            << "[SMM] t0 = " << hp.t0 << endl
            << "[SMM] p = " << hp.p << " (q = " << _q << ")" << endl
            << "[SMM] lambda = " << hp.lambda << endl
            << "[SMM] weight init. gaussian std. = " << hp.gauss_init_std << endl
	    << "[SMM] NBNL mode: " << _nbnl_mode << endl
            << "[SMM] offload models dev <-> host = " << _offload_num << endl;

    if (_save_every_t) {
        __log__ << "Will save models to " << _model_dir << " every " << _save_every_t << " minibatches." << endl;
    }

    if (_validation_set) {
        __log__ << "Validation set supplied. Will run over every " << _save_every_t << " minibatches." << endl;
    }

    float prev_validation_score = 0.0f;
    Mat scores(minibatch_size, classes);
    vector<CuMat> score_grads;
    Vec softmax_scores;
    try_init_models(classes, dim, hp.n);
    init_random_score_grads(score_grads, classes, minibatch_size);

    __log__ << "M-b.\tAER\tsec" << endl;

    Timer wallclock;
    wallclock.start();
    _global_timer.reset();


    // experimental
    float prev_aer = 0.0f;

    do {
        check_minibatch(mb);

        gamma_t = sqrt((1.0f + hp.t0) / (_t + hp.t0));

        move_W_bar_to_host();
        move_G_to_host();

        // Loading minibatch to device
        CuMat cuX(_handle);
        cuX.from_eigen(*(mb.X));

        _global_timer.start();

        log_sum_softmax = compute_class_scores(scores, score_grads, cuX, *(mb.y));
        loss = log_sum_softmax/minibatch_size;
        _sum_loss += loss;

        _global_timer.pause();

        if (is_pow_of_two(_t)) {
            wallclock.pause();
            __log__ << _t << '\t' << setprecision(3) << _sum_loss/_t << '\t' << float(wallclock.get_ms()) / 1000 << endl;
            wallclock.start();

            // experimental: tuning L by looking at AER diff
            // float curr_aer = _sum_loss/_t;
            // if (fabs(curr_aer - prev_aer) < 5) {
            //     _hp.L *= 2;
            //     W_bar_const = _hp.L / (_hp.L + hp.lambda);
            //     G_const = 1.0f / (_hp.L + hp.lambda);

            //     if (is_pow_of_two(_t)) {
            //         cerr << "Turning L by AER tracking:: Adjusting L. New L = " << _hp.L << endl;
            //     }
            // }
            // prev_aer = curr_aer;
        }

        if (_record_aer)
            _aer_track.push_back(_sum_loss/_t);

        move_G_to_dev();

        _global_timer.start();

        for (size_t k = 0; k < classes; ++k) {
            softmax_scores = scores.col(k);

            // if (_t <= _hp.pass_length) {
            //     update_G_convexified(_G[k], score_grads[k], prev_score_grads[k], cuX, *(mb.y), softmax_scores, gamma_t, k);
            //     prev_score_grads[k] = score_grads[k];
            // }
            // else {
            //     update_G(_G[k], score_grads[k], cuX, *(mb.y), softmax_scores, gamma_t, k);
            // }

            update_G(_G[k], score_grads[k], cuX, *(mb.y), softmax_scores, gamma_t, k);
        }

#ifdef NDEBUG
        // Freeing up device memory
        cuX.dealloc();
        score_grads.clear();
#endif

        // Continuing to update parameters
        move_W_bar_to_dev();

        _W_norm_sum = 0;
        for (size_t k = 0; k < classes; ++k) {

            _W_bar[k].set_add(_W[k], gamma_t, 1.0f - gamma_t);
            _W[k].set_add(_W_bar[k], _G[k], W_bar_const, -G_const);

            _W_norm_sum += _W[k].trace_mTm(_W[k]);
        }

        _global_timer.pause();

        // cerr << "Timers:" << endl;
        // cerr << "\tcompute_class_scores = " << timer_compute_class_scores.get_string() << endl;
        // cerr << "\tupdates = " << timer_updates.get_string() << endl;
        // cerr << "\tcuX = " << timer_cuX.get_string() << endl;

        if (_save_every_t && ((_t % _save_every_t) == 0)) {
            save();

            if (_validation_set) {
                cerr << "Running on a validation set..." << endl;
		cerr << "-----------------------------------------------------" << endl;
                float validation_score = this->test(*_validation_set, true);
                _validation_set->reset();
		cerr << "-----------------------------------------------------" << endl;
                cerr << "Done." << endl;
                cerr << "Prev. validation score = " << prev_validation_score << endl;
                cerr << "Validation score = " << validation_score << endl;

                // decision
                // if (fabs(validation_score - prev_validation_score) < 0.1) {
                //     _hp.L *= 2;
                //     W_bar_const = _hp.L / (_hp.L + hp.lambda);
                //     G_const = 1.0f / (_hp.L + hp.lambda);

                //     cerr << "Adjusting L. New L = " << _hp.L << endl;
                // }

                prev_validation_score = validation_score;
            }

        }

        _t++;
    }
    while (example_set.get_minibatch(mb));

    _last_train_duration_ms = _global_timer.get_ms();

    if (_save_every_t)
        save();
}

void CuSmoothML3::resume(const uint32_t classes, MinibatchExampleSet &example_set) {
    _W_norm_sum = 0.0f;

    float W_bar_const = _hp.L / (_hp.L + _hp.lambda);
    float G_const = 1.0f / (_hp.L + _hp.lambda);

    Minibatch mb;
    CuMat score_grad(_handle, _hp.n, classes);
    float log_sum_softmax;
    float loss;
    float gamma_t = 1.0f;

    bool got_example = example_set.get_minibatch(mb);
    if (!got_example)
        return;

    const size_t minibatch_size = mb.X->cols();
    const size_t dim = mb.X->rows();

    __log__ << "[SMM] Starting training with " << classes << " classes." << endl
            << "[SMM] minibatch size = " << minibatch_size << endl
            << "[SMM] dimension = " << dim << endl
            << "[SMM] n = " << _hp.n << endl
            << "[SMM] L = " << _hp.L << endl
            << "[SMM] t0 = " << _hp.t0 << endl
            << "[SMM] p = " << _hp.p << " (q = " << _q << ")" << endl
            << "[SMM] lambda = " << _hp.lambda << endl
            << "[SMM] weight init. gaussian std. = " << _hp.gauss_init_std << endl
            << "[SMM] offload models dev <-> host = " << _offload_num << endl
	    << "[SMM] RESUME INFO:" << endl
	    << "[SMM] t = " << _t << endl
	    << "[SMM] AER = " << (float)_sum_loss/_t << endl;

    if (_save_every_t) {
        __log__ << "Will save models to " << _model_dir << " every " << _save_every_t << " minibatches." << endl;
    }

    if (_validation_set) {
        __log__ << "Validation set supplied. Will tune L every " << _save_every_t << " minibatches." << endl;
    }

    float prev_validation_score = 0.0f;
    Mat scores(minibatch_size, classes);
    vector<CuMat> score_grads;
    Vec softmax_scores;
    init_random_score_grads(score_grads, classes, minibatch_size);

    __log__ << "M-b.\tAER\tsec" << endl;

    Timer wallclock;
    wallclock.start();


    _t++;
    size_t restarted_t = 0;
    float restarted_sum_loss = 0.0f;
    do {
        check_minibatch(mb);

        gamma_t = sqrt((1.0f + _hp.t0) / (_t + _hp.t0));

        move_W_bar_to_host();
        move_G_to_host();

        // Loading minibatch to device
        CuMat cuX(_handle);
        cuX.from_eigen(*(mb.X));

        _global_timer.start();

        log_sum_softmax = compute_class_scores(scores, score_grads, cuX, *(mb.y));
        loss = log_sum_softmax/minibatch_size;
        _sum_loss += loss;
	restarted_sum_loss += loss;

        _global_timer.pause();

        if (is_pow_of_two(_t)) {
            wallclock.pause();
            __log__ << _t << '\t' << setprecision(3) << _sum_loss/_t << '\t' << float(wallclock.get_ms()) / 1000 << endl;
            wallclock.start();
        }

	if (is_pow_of_two(restarted_t)) {
            wallclock.pause();
            __log__ << "Since resume: " <<  restarted_t << '\t' << setprecision(3) << restarted_sum_loss/restarted_t << '\t' << float(wallclock.get_ms()) / 1000 << endl;
            wallclock.start();
        }

        if (_record_aer)
            _aer_track.push_back(_sum_loss/_t);

        move_G_to_dev();

        _global_timer.start();

        for (size_t k = 0; k < classes; ++k) {
            softmax_scores = scores.col(k);
            update_G(_G[k], score_grads[k], cuX, *(mb.y), softmax_scores, gamma_t, k);
        }

#ifdef NDEBUG
        // Freeing up device memory
        cuX.dealloc();
        score_grads.clear();
#endif

        // Continuing to update parameters
        move_W_bar_to_dev();

        _W_norm_sum = 0;
        for (size_t k = 0; k < classes; ++k) {

            _W_bar[k].set_add(_W[k], gamma_t, 1.0f - gamma_t);
            _W[k].set_add(_W_bar[k], _G[k], W_bar_const, -G_const);

            _W_norm_sum += _W[k].trace_mTm(_W[k]);
        }

        _global_timer.pause();

        if (_save_every_t && ((_t % _save_every_t) == 0)) {
            save();

            if (_validation_set) {
                cerr << "Running on a validation set..." << endl;
                float validation_score = this->test(*_validation_set, true);
                _validation_set->reset();
                cerr << "Done." << endl;
                cerr << "Prev. validation score = " << prev_validation_score << endl;
                cerr << "Validation score = " << validation_score << endl;

                // decision
                if (fabs(validation_score - prev_validation_score) < 0.1) {
                    _hp.L *= 2;
                    W_bar_const = _hp.L / (_hp.L + _hp.lambda);
                    G_const = 1.0f / (_hp.L + _hp.lambda);

                    cerr << "Adjusting L. New L = " << _hp.L << endl;
                }

                prev_validation_score = validation_score;
            }

        }

        _t++;
	restarted_t++;
    }
    while (example_set.get_minibatch(mb));

    _last_train_duration_ms = _global_timer.get_ms();

    if (_save_every_t)
        save();
}

float CuSmoothML3::compute_surrogate(const float loss, vector<CuMat> &W_t_1, const vector<CuMat> &score_grads, const Mat &softmax_scores,
                                     const CuMat &cuX, const vector<uint32_t> &y) {
    const size_t minibatch_size = cuX._cols;
    const size_t dim = cuX._rows;
    float surrogate = loss;
    CuMat A(_handle, dim, _hp.n);
    CuMat B(_handle);
    CuMat neg_class_ip(_handle, dim, _hp.n);
    CuVec d_weights(_handle, minibatch_size);

    for (size_t i = 0; i < _W.size(); ++i) {
        A.set_add(_W[i], W_t_1[i], 1.0f, -1.0f);
        surrogate += _hp.L/2.0f * A.trace_mTm(A);
        surrogate += _hp.lambda/2.0f * _W[i].trace_mTm(_W[i]);

        CuMat cpy_score_grads;
        cpy_score_grads = score_grads[i];
        d_weights.from_eigen(softmax_scores.col(i));
        cpy_score_grads.set_dmm(d_weights, false);
        neg_class_ip.set_mmT(cuX, cpy_score_grads, 1.0f, 0.0f);

        surrogate += neg_class_ip.trace_mTm(A) / minibatch_size;
    }

    CuVec vx(_handle, dim), x(_handle, dim);
    size_t k;

    for (size_t i = 0; i < minibatch_size; ++i) {
        k = y[i];
        A.set_add(_W[k], W_t_1[k], 1.0f, -1.0f);

        score_grads[k].get_col(vx, i);
        cuX.get_col(x, i);

        B.outer(x, vx);
        assert((B.to_eigen() - x.to_eigen() * vx.to_eigen().transpose()).norm() <= 1.0e-5);
        surrogate -= B.trace_mTm(A) / minibatch_size;
    }

    return surrogate;
}

float CuSmoothML3::test(MinibatchExampleSet &example_set, const bool on_gpu) {
    Minibatch mb;
    size_t correct = 0;
    size_t total = 0;

    Timer test_timer;

    cerr << "Testing on " << (on_gpu ? "GPU" : "CPU") << " in " << ((!_nbnl_mode) ? "standard" : "NBNL") << " mode." << endl;
    __log__ << "M-b.\tAER\tsec\tcorr.\ttotal" << endl;

    uint32_t nbnl_examples = 0;
    size_t mb_i = 0;
    while (example_set.get_minibatch(mb)) {
        check_minibatch(mb);
	if (_nbnl_mode && (mb.tags->empty())) {
	    throw invalid_argument("Testing in NBNL mode requires <tags> field in minibatch indicating beginning of meta-examples (images) by 1 and rest by 0.");
	}

        test_timer.start();

        if (on_gpu) {
	    nbnl_examples = 0;
            correct += num_correct(mb, _nbnl_mode, &nbnl_examples);

	    if (_nbnl_mode)
		total += nbnl_examples;
	    else
		total += mb.y->size();

	    //cerr << "test:: minibatch contains " << nbnl_examples << " examples." << endl;
        }
        else {
	    nbnl_examples = 0;
            correct += num_correct_cpu(*(mb.X), *(mb.y), _nbnl_mode, &nbnl_examples);
	    if (_nbnl_mode)
		total += nbnl_examples;
	    else
		total += mb.y->size();
        }

        test_timer.pause();

	mb_i++;
	if (is_pow_of_two(mb_i)) {
	    __log__ << mb_i << '\t' << setprecision(3) << 1.0f-float(correct)/float(total) << '\t' << float(test_timer.get_ms()) / 1000
		    << '\t' << correct << "\t" << total << endl;
	}
    }

    cerr << "correct = " << correct << endl
	 << "total = " << total << endl;

    _last_test_duration_ms = test_timer.get_ms();

    // string W_file = "/idiap/temp/ikuzbor/W_orig.txt";
    // cerr << "DEBUG:: Dumping weight matrices to " << W_file << endl;
    // ofstream out(W_file.c_str(), ios::out | ios::trunc);
    // for (size_t i = 0; i < _W.size(); ++i) {
    // 	out << _W[i].to_eigen() << endl;
    // 	out << "-----------------------------" << endl;
    // }

    return float(correct)/float(total);
}

inline void CuSmoothML3::try_init_models(const uint32_t classes, const uint32_t dim, const uint32_t n) {
    __log__ << "Creating dense randomly initialized " << n << " models of dimension " << dim << " for each class." << endl;

    if (_W.empty()) {
        _W.resize(classes);
        _W_norm_sum = 0;

        for (uint32_t i = 0; i < classes; ++i) {
            Mat W_init_rand(dim, n);
            for (size_t j = 0; j < W_init_rand.size(); ++j)
                W_init_rand.data()[j] = rand_gauss()*_hp.gauss_init_std;

            // W_init_rand.setRandom();
            // W_init_rand *= 0.01;

            _W[i]._handle = _handle;
            _W[i].from_eigen(W_init_rand);

            _W_norm_sum += _W[i].trace_mTm(_W[i]);
        }
    }

    _W_bar.resize(classes);
    _G.resize(classes);

    Mat W_init_zero(dim, n);
    W_init_zero.setZero();

    for (uint32_t i = 0; i < classes; ++i) {
        _W_bar[i]._handle = _handle;
        _W_bar[i].from_eigen(W_init_zero);

        _G[i]._handle = _handle;
        _G[i].from_eigen(W_init_zero);
    }
}

void CuSmoothML3::move_W_bar_to_host() {
    if (_offload_num < 200)
        return;

    for (size_t i = 0; i < _offload_num; ++i) {
        _W_bar[i].move_to_host();
    }
}

void CuSmoothML3::move_W_bar_to_dev() {
    if (_offload_num < 200)
        return;

    for (size_t i = 0; i < _offload_num; ++i) {
        _W_bar[i].move_to_device();
    }
}

void CuSmoothML3::move_G_to_host() {
    if (_offload_num < 200)
        return;

    for (size_t i = 0; i < _offload_num; ++i) {
        _G[i].move_to_host();
    }
}

void CuSmoothML3::move_G_to_dev() {
    if (_offload_num < 200)
        return;

    for (size_t i = 0; i < _offload_num; ++i) {
        _G[i].move_to_device();
    }
}

inline void CuSmoothML3::check_minibatch(const Minibatch &mb) {
    uint32_t _max;
    _max = *max_element(mb.y->begin(), mb.y->end());

    if (_max >= _W.size())
        __warn__ << "Got example with label " << _max << ". Labels must not exceed " << _W.size()-1 << "!" << endl;
}

size_t CuSmoothML3::num_correct(const Minibatch &mb, const bool nbnl, uint32_t *nbnl_examples) {
    CuMat cuX(_handle);
    cuX.from_eigen(*(mb.X));
    const vector<uint32_t> &y = *(mb.y);
    
    const size_t m = cuX._cols;
    const size_t classes = _W.size();

    CuMat prod(_handle);

    CuVec ones(_handle);
    ones.from_eigen(Vec::Ones(_hp.n));

    CuVec t_scores(_handle);
    Mat scores(m, classes);

    Vec v;

    scores.resize(m, classes);

    for (uint32_t i = 0; i < classes; ++i) {
        prod.set_mTm(_W[i], cuX, 1.0f, 0.0f); // n x m
        prod.nonneg_and_pow(_q);

        t_scores.set_mv(prod, ones, true); // m
        t_scores.pow(1.0f/_q); // [ ||[W x_1]_+||_q, ..., ||[W x_m]_+||_q ]

        t_scores.to_eigen(v);
        scores.col(i) = v;
    }

    Vec::Index max_ix = 0;
    size_t total = 0;

    if (!nbnl) {
	for (uint32_t i = 0; i < m; ++i) {
	    scores.row(i).maxCoeff(&max_ix);
	    total += size_t(max_ix == y[i]);
	    // cerr << "Correct: " << y[i] << endl
	    // 	 << "Pred: " << max_ix << endl;
	    // cerr << "Length: " << scores.row(i).size() << endl;
	    //cerr << scores.row(i) << endl;
	}
    }
    else {
	const vector<uint32_t> &example_start = *(mb.tags);
	Vec t_scores(classes);
	t_scores.setZero();
	uint32_t curr_label = y[0];
	size_t tmp = 0;
	(*nbnl_examples) = 0;

	//cerr << "num_correct:: m = " << m << endl;
	for (uint64_t i = 0; i < m; ++i) {
	    if (example_start[i] && (i == 0)) {
		curr_label = y[i];
	    }

	    //cerr << "num_correct:: i = " << i << "; example_start[i] = " << example_start[i] << endl;
		
	    if (example_start[i] && (i > 0)) {
		// Example finished, so we are evaluating
		t_scores.maxCoeff(&max_ix);
		// cerr << t_scores.transpose() << endl;
		total += size_t(max_ix == curr_label);
		(*nbnl_examples)++;
		// Resetting scores
		t_scores = scores.row(i);
		
		// cerr << t_scores.transpose() << endl;
		// cerr << "Current label: " << curr_label << endl
		//      << "NBNL pred.: " << max_ix << endl;
		//      << "Correct: " << total << endl
		//      << "Total: " << *nbnl_examples << endl;
		
		//cerr << "NBNL example: " << (*nbnl_examples) << "; instances: " << tmp << endl;
		//tmp = 1;
		curr_label = y[i];
	    }
	    
	    if (!example_start[i]) {
		//cerr << scores.row(i) << endl;
		t_scores += scores.row(i);
		//tmp++;
	    }
	}

	t_scores.maxCoeff(&max_ix);
	total += size_t(max_ix == curr_label);
	(*nbnl_examples)++;
    }

    return total;
}

size_t CuSmoothML3::num_correct_cpu(const Mat &X, const vector<uint32_t> &y, const bool nbnl, uint32_t *nbnl_examples) {
    const size_t m = X.cols();
    const size_t classes = _W.size();

    assert(m == y.size());
    if (m == 0)
	return 0;

    Mat prod;
    Vec ones = Vec::Ones(_hp.n);

    Vec t_scores;
    Mat scores(m, classes);

    Vec v;

    scores.resize(m, classes);

    for (uint32_t i = 0; i < classes; ++i) {
        //cerr << "num_correct_cpu::class " << i << endl;
        prod = _W[i]._tmp_storage.transpose() * X;
        prod = prod.array().max(0).pow(_q).matrix();

        t_scores = prod.transpose() * ones; // m
        t_scores = t_scores.array().pow(1.0f/_q).matrix(); // [ ||[W x_1]_+||_q, ..., ||[W x_m]_+||_q ]

        scores.col(i) = t_scores;

    }

    Vec::Index max_ix = 0;
    size_t total = 0;
    size_t tmp = 0;

    if (!nbnl) {
	for (uint32_t i = 0; i < m; ++i) {
	    scores.row(i).maxCoeff(&max_ix);
	    total += size_t(max_ix == y[i]);
	}
    }
    else {
	(*nbnl_examples) = 0;

	uint32_t curr_label = y[0];
	t_scores = Vec::Zero(classes);
	size_t tmp = 0;
	for (uint32_t i = 0; i < m; ++i) {
	    if (curr_label != y[i]) {
		t_scores.maxCoeff(&max_ix);
		total += size_t(max_ix == curr_label);

		t_scores = Vec::Zero(classes);
		curr_label = y[i];

		t_scores = scores.row(i);
		(*nbnl_examples)++;

		//cerr << "NBNL example: " << (*nbnl_examples) << "; instances: " << tmp << endl;
		//tmp = 1;
	    }
	    else {
		t_scores += scores.row(i);
		//tmp++;
	    }
	}

	t_scores.maxCoeff(&max_ix);
	total += size_t(max_ix == curr_label);
	(*nbnl_examples)++;
    }

    return total;
}


float CuSmoothML3::compute_class_scores(Mat &scores, vector<CuMat> &score_grads, const CuMat &cuX, const vector<uint32_t> &y, const bool dont_compute_grad) const {
    const size_t m = cuX._cols;
    const size_t classes = _W.size();

    CuMat prod(_handle);
    //CuMat grad(_handle);

    CuVec ones(_handle);
    ones.from_eigen(Vec::Ones(_hp.n));

    CuVec t_scores(_handle);
    CuVec cpy_t_scores(_handle);

    Vec v;

    scores.resize(m, classes);

    if (!dont_compute_grad)
        score_grads.resize(classes);

    for (uint32_t i = 0; i < classes; ++i) {
        prod.set_mTm(_W[i], cuX, 1.0f, 0.0f); // n x m

        CuMat &grad = score_grads[i];

        if (!dont_compute_grad) {
            grad = prod;
        }

        prod.nonneg_and_pow(_q);

        t_scores.set_mv(prod, ones, true); // m
        t_scores.pow(1.0f/_q); // [ ||[W x_1]_+||_q, ..., ||[W x_m]_+||_q ]

        t_scores.to_eigen(v);
        scores.col(i) = v;

        if (!dont_compute_grad) {
            grad.set_dmm(t_scores.inv_nz(), false);
            grad.nonneg_and_pow(_q - 1.0f);
        }
    }

    float Z = 0.0f;
    float yt_score = 0.0f; // needed for loss computation

    // Computing softmax here
    for (uint32_t i = 0; i < m; ++i) {
        yt_score = scores(i, y[i]);
        t_scores.from_eigen(scores.row(i));

        float max_score = t_scores.amax();

        cpy_t_scores = t_scores;

        t_scores.sub_and_exp(max_score);
        float logZ = logf(t_scores.asum());

        cpy_t_scores.sub_and_exp(max_score + logZ);

        cpy_t_scores.to_eigen(v);
        scores.row(i) = v;

        //cerr << max_score << ", " << logZ << ", " << yt_score << endl;
        Z += max_score + logZ - yt_score;
    }


    return Z;
}

inline void CuSmoothML3::update_G(CuMat &G, const CuMat &score_grads, const CuMat &cuX, const vector<uint32_t> &y,
                                  const Vec &softmax_scores,
                                  const float gamma_t, const size_t k) {
    const size_t minibatch_size = cuX._cols;
    const size_t minibatch_size_n = cuX._cols;
    Vec pos_class_weights(minibatch_size);
    CuMat cpy_score_grads = score_grads;
    CuVec d_weights(_handle);
    d_weights.from_eigen(softmax_scores);

    // cerr << "*************************************" << endl;
    // cerr << score_grads.to_eigen() << endl;
    // cerr << "*************************************" << endl;

    cpy_score_grads.set_dmm(d_weights, false);
    G.set_mmT(cuX, cpy_score_grads, gamma_t / minibatch_size_n, 1.0f - gamma_t / minibatch_size_n);
    // cerr << G.to_eigen() << endl;
    // cerr << "---------------------------------" << endl;

    // Copying second time to subtract positive class gradients
    cpy_score_grads = score_grads;

    for (size_t i = 0; i < minibatch_size; ++i) {
        if (y[i] == k)
            pos_class_weights[i] = 1.0f;
        else
            pos_class_weights[i] = 0.0f;
    }

    // cerr << "+++++++++++++++++++++++++++++++++++++" << endl;
    // cerr << pos_class_weights.transpose() << endl;
    // cerr << "+++++++++++++++++++++++++++++++++++++" << endl;

    d_weights.from_eigen(pos_class_weights);
    cpy_score_grads.set_dmm(d_weights, false);


    // cerr << d_weights.to_eigen().transpose() << endl;

    G.set_mmT(cuX, cpy_score_grads, -gamma_t / minibatch_size_n, 1.0f);

    // cerr << G.to_eigen() << endl;
    // cerr << "minibatch_size = " << minibatch_size << endl;
}

void CuSmoothML3::init_random_score_grads(vector<CuMat> &score_grads, const size_t classes, const size_t minibatch_size) {
    score_grads.resize(classes);

    CuMat prod(_handle);
    Mat init(_hp.n, minibatch_size);
    CuVec t_scores(_handle);
    CuVec ones(_handle);
    ones.from_eigen(Vec::Ones(_hp.n));

    for (size_t i = 0; i < score_grads.size(); ++i) {
        init.setRandom();
        prod.from_eigen(init);

        CuMat &grad = score_grads[i];
        grad = prod;

        prod.nonneg_and_pow(_q);

        t_scores.set_mv(prod, ones, true); // m
        t_scores.pow(1.0f/_q); // [ ||[W x_1]_+||_q, ..., ||[W x_m]_+||_q ]

        grad.set_dmm(t_scores.inv_nz(), false);

        grad.nonneg_and_pow(_q - 1.0f);
    }
}

void CuSmoothML3::set_storage_params(const std::string &path, const size_t save_every_t, const std::string model_tag) {
    _model_dir = path;
    _model_tag = model_tag;
    _save_every_t = save_every_t;
}

void CuSmoothML3::save() {
    assert(!_model_dir.empty());

    stringstream ss;

    ss << "cusmoothml3_model_"
       << _model_tag << "_"
       << "p=" << _hp.p << "_"
       << "n=" << _hp.n << "_"
       << "lambda=" << _hp.lambda << "_"
       << "t0=" << _hp.t0 << "_"
       << "L=" << _hp.L_start << "_"
       << "t=" << _t
       << ".bin";
    const string filename = ss.str();
    const string path = _model_dir + "/" + filename;

    cerr << "Saving model file to " << path << endl;

    ofstream out(path.c_str(), ios::out | ios::binary | ios::trunc);
    out << __MAGIC__;

    size_t classes = _W.size();
    out.write((char*) (&classes), sizeof(classes));

    for (size_t i = 0; i < _W.size(); ++i) {
        _W[i].save(out);
        _G[i].save(out);
        _W_bar[i].save(out);
    }

    out.write((char*) (&_hp), sizeof(_hp));
    out.write((char*) (&_t), sizeof(_t));
    out.write((char*) (&_sum_loss), sizeof(_sum_loss));
    out.write((char*) (&_q), sizeof(_q));

    out.write((char*) (&_offload_num), sizeof(_offload_num));
    out.write((char*) (&_record_aer), sizeof(_record_aer));

    size_t track_size = _aer_track.size();
    out.write((char*) (&track_size), sizeof(track_size));
    for (std::list<float>::iterator it=_aer_track.begin();
         it != _aer_track.end(); ++it) {
        float v = *it;
        out.write((char*) (&v), sizeof(v));
    }

    _last_train_duration_ms = _global_timer.get_ms();
    out.write((char*) (&_last_train_duration_ms), sizeof(_last_train_duration_ms));

    _global_timer.save(out);

    out.close();
    cerr << "Done." << endl;
}

void CuSmoothML3::load(const std::string &filename, const bool to_gpu) {
    cerr << "Trying to load model from file " << filename << endl;
    ifstream in(filename.c_str(), ios::in | ios::binary);

    char magic[__MAGIC_LEN__+1] = {0};
    in.read((char*) (&magic), __MAGIC_LEN__);

    cerr << magic << endl;

    if (strcmp(magic, __MAGIC__)) {
        cerr << "Could not load. File is incorrect." << endl;
        return;
    }

    _handle = init_cublas();
    //cudaCheckErrors("init_cublas()");

    size_t classes;
    in.read((char*) (&classes), sizeof(classes));

    _W.resize(classes);
    //std::cerr << "_G and _W_bar compatibility issue!!!" << endl;
    _G.resize(classes);
    _W_bar.resize(classes);

    for (size_t i = 0; i < _W.size(); ++i) {
        _W[i].load(in, to_gpu);
        _G[i].load(in);
        _W_bar[i].load(in);

        _W[i]._handle = _handle;
        _G[i]._handle = _handle;
        _W_bar[i]._handle = _handle;
    }

    in.read((char*) (&_hp), sizeof(_hp));
    in.read((char*) (&_t), sizeof(_t));
    in.read((char*) (&_sum_loss), sizeof(_sum_loss));
    in.read((char*) (&_q), sizeof(_q));

    in.read((char*) (&_offload_num), sizeof(_offload_num));
    in.read((char*) (&_record_aer), sizeof(_record_aer));

    size_t track_size;
    in.read((char*) (&track_size), sizeof(track_size));

    for (size_t i = 0; i < track_size; ++i) {
        float v;
        in.read((char*) (&v), sizeof(v));
        _aer_track.push_back(v);
    }

    in.read((char*) (&_last_train_duration_ms), sizeof(_last_train_duration_ms));

    _global_timer.load(in);

    in.close();
    cerr << "Done." << endl;

    cerr << "CuSmoothML3::load\t" << "_W.size() = " << _W.size() << endl
	 << "CuSmoothML3::load\t" << "_q = " << _q << endl;

    // string W_file = "/idiap/temp/ikuzbor/W_restored.txt";
    // cerr << "DEBUG:: Dumping weight matrices to " << W_file << endl;
    // ofstream out2(W_file.c_str(), ios::out | ios::trunc);
    // for (size_t i = 0; i < _W.size(); ++i) {
    // 	out2 << _W[i].to_eigen() << endl;
    // 	out2 << "-----------------------------" << endl;
    // }
}

void CuSmoothML3::adagrad_on_convexified(MinibatchExampleSet &example_set, const size_t classes, const size_t iter, float learning_rate) {
    Minibatch mb;

    learning_rate = 1;

    bool got_example = example_set.get_minibatch(mb);
    if (!got_example)
        return;

    const size_t minibatch_size = mb.X->cols();
    const size_t dim = mb.X->rows();
    const size_t n = _hp.n;

    __log__ << "[AdaGrad] Starting training with " << classes << " classes." << endl
            << "[AdaGrad] minibatch size = " << minibatch_size << endl
            << "[AdaGrad] dimension = " << dim << endl
            << "[AdaGrad] n = " << n << endl
            << "[AdaGrad] p = " << _hp.p << " (q = " << _q << ")" << endl
            << "[AdaGrad] learning rate = " << learning_rate << endl
            << "[AdaGrad] lambda = " << _hp.lambda << endl;

    cerr << "AdaGrad:: initializing parameters." << endl;
    // Creating parameters
    _W.resize(classes);
    vector<CuMat> G_bar(classes);

    Mat W_init_ones = 1e-8 * Mat::Ones(dim, n);

    for (uint32_t i = 0; i < classes; ++i) {
        Mat W_init_rand(dim, n);
        W_init_rand.setRandom();

        _W[i]._handle = _handle;
        _W[i].from_eigen(W_init_rand);

        G_bar[i]._handle = _handle;
        G_bar[i].from_eigen(W_init_ones);
    }

    Mat scores(minibatch_size, classes);
    vector<CuMat> score_grads(classes);
    vector<CuMat> prev_score_grads(classes);
    float log_sum_softmax;
    CuMat G(_handle, dim, n), cpy_G(_handle, dim, n);
    float loss = 0.0f;

    init_random_score_grads(score_grads, classes, minibatch_size);
    copy(score_grads.begin(), score_grads.end(), prev_score_grads.begin());

    do {
        check_minibatch(mb);

        // Loading minibatch to device
        CuMat cuX(_handle);
        cuX.from_eigen(*(mb.X));

        _global_timer.start();

        // Computing scores and score gradients
        log_sum_softmax = compute_class_scores(scores, score_grads, cuX, *(mb.y));
        loss = log_sum_softmax/minibatch_size;
        _sum_loss += loss;

        _global_timer.pause();

        if (is_pow_of_two(_t)) {
            __log__ << _t << '\t' << setprecision(3) << _sum_loss/_t << "\t?\t[AdaGrad]" << endl;
        }

        if (_record_aer)
            _aer_track.push_back(_sum_loss/_t);

        _global_timer.start();

        // Computing gradient
        for (size_t k = 0; k < classes; ++k) {
            compute_convexified_G(G, cuX,
                                  scores.col(k),
                                  prev_score_grads[k],
                                  prev_score_grads[k],
                                  *(mb.y), k);

            //prev_score_grads[k] = score_grads[k];

            // cerr << "-----------------------------------------------------------" << endl;
            // cerr << "t = " << _t << "; k = " << k << endl;
            // cerr << "G:" << endl << G << endl;

            cpy_G = G;
            cpy_G.pow(2.0f);
            G_bar[k].set_add(G_bar[k], cpy_G, 1.0f, 1.0f);

            // cerr << "G_bar:" << endl << G_bar[k] << endl;

            cpy_G = G_bar[k];
            cpy_G.pow(-0.5f);

            // cerr << "1/sqrt(G_bar):" << endl << cpy_G << endl;

            // cerr << "G:" << endl << G << endl;

            cpy_G.set_hadamard(G);
            // cerr << "G o 1/sqrt(G_bar):" << endl << cpy_G << endl;

            // cerr << "[eigen] _W[k]^(t+1) = " << endl << _W[k].to_eigen() - cpy_G.to_eigen() * learning_rate << endl;

            _W[k].set_add(_W[k], cpy_G, 1.0f, -learning_rate);

            // cerr << "_W[k]^(t+1) = " << endl << _W[k] << endl;
        }

        _global_timer.pause();

        if (_t == iter) {
            return;
        }

        _t++;
    }
    while (example_set.get_minibatch(mb));
}

void CuSmoothML3::pegasos_on_convexified(MinibatchExampleSet &example_set, const size_t classes, const size_t iter) {
    Minibatch mb;

    bool got_example = example_set.get_minibatch(mb);
    if (!got_example)
        return;

    const size_t minibatch_size = mb.X->cols();
    const size_t dim = mb.X->rows();
    const size_t n = _hp.n;

    __log__ << "[Pegasos] Starting training with " << classes << " classes." << endl
            << "[Pegasos] minibatch size = " << minibatch_size << endl
            << "[Pegasos] dimension = " << dim << endl
            << "[Pegasos] n = " << n << endl
            << "[Pegasos] p = " << _hp.p << " (q = " << _q << ")" << endl
            << "[Pegasos] lambda = " << _hp.lambda << endl;

    cerr << "Pegasos:: initializing parameters." << endl;
    // Creating parameters
    _W.resize(classes);
    vector<CuMat> G_bar(classes);

    Mat W_init_ones = 1e-8 * Mat::Ones(dim, n);

    for (uint32_t i = 0; i < classes; ++i) {
        Mat W_init_rand(dim, n);
        W_init_rand.setRandom();

        _W[i]._handle = _handle;
        _W[i].from_eigen(W_init_rand);
    }

    Mat scores(minibatch_size, classes);
    vector<CuMat> score_grads(classes);
    vector<CuMat> prev_score_grads(classes);
    float log_sum_softmax;
    CuMat G(_handle, dim, n);
    float loss = 0.0f;

    init_random_score_grads(score_grads, classes, minibatch_size);
    copy(score_grads.begin(), score_grads.end(), prev_score_grads.begin());

    do {
        check_minibatch(mb);

        // Loading minibatch to device
        CuMat cuX(_handle);
        cuX.from_eigen(*(mb.X));

        _global_timer.start();

        // Computing scores and score gradients
        log_sum_softmax = compute_class_scores(scores, score_grads, cuX, *(mb.y));
        loss = log_sum_softmax/minibatch_size;
        _sum_loss += loss;

        _global_timer.pause();

        if (is_pow_of_two(_t)) {
            __log__ << _t << '\t' << setprecision(3) << _sum_loss/_t << "\t?\t[Pegasos]" << endl;
        }

        if (_record_aer)
            _aer_track.push_back(_sum_loss/_t);

        _global_timer.start();

        // Computing gradient
        for (size_t k = 0; k < classes; ++k) {
            compute_convexified_G(G, cuX,
                                  scores.col(k),
                                  score_grads[k],
                                  prev_score_grads[k],
                                  *(mb.y), k);

            prev_score_grads[k] = score_grads[k];

            _W[k].set_add(_W[k], G, 1.0f - 1.0f/_t, 1.0f/(_hp.lambda * _t));
            G = _W[k];

            // Projection step
            float norm = sqrtf(G.trace_mTm(G));
            float scale = (1.0f / sqrtf(_hp.lambda)) / norm;
            if (scale < 1.0f)
                _W[k].scale(scale);
        }

        _global_timer.pause();

        if (_t == iter) {
            return;
        }

        _t++;
    }
    while (example_set.get_minibatch(mb));
}

// void CuSmoothML3::accelerated_minibatch_descent(const CuMat &cuX, const vector<uint32_t> &y, const CuMat &score_grads) {
//     // Implements one step of Nesterov's second method
//     // Basic setting of schedule (no line search)
//     const size_t classes = _W.size();
//     const size_t dim = cuX._rows;

//     const float theta_t = 2.0f / (_t + 1.0f);
//     const float eta_t = 1.0f / _hp.L;

//     CuMat V(_handle), G(_handle);
//     vector<CuMat> score_grads(classes);
//     vector<CuMat> Z(classes);

//     for (size_t k = 0; k < classes; ++k) {
//         Z[k]._handle = _handle;
//         Z[k]._rows = dim;
//         Z[k]._cols = _hp.n;
//         Z[k].alloc();
//     }

//     compute_class_scores(scores, score_grads, Z, cuX, y);

//     for (size_t k = 0; k < _W.size(); ++k) {
//         V = _W[k];

//         Z[k].set_add(_W[k], V, 1.0f - theta_t, theta_t);
//         compute_convexified_G(G, cuX, score_grads, y, k);
//         V.set_add(V, G, 1.0f, -eta_t / theta_t);

//         norm = sqrtf(V.trace_mTm(V));
//         if (norm > (eta_t / theta_t) * tau)
//             V.scale(1.0f / norm);

//         //_W[k].set_add
//     }
// }

inline void CuSmoothML3::compute_convexified_G(CuMat &G, const CuMat &cuX,
                                               const Vec &softmax_scores,
                                               const CuMat &score_grads,
                                               const CuMat &pos_score_grads,
                                               const vector<uint32_t> &y, const uint32_t k) {
    const size_t minibatch_size = cuX._cols;
    Vec pos_class_weights(minibatch_size);
    CuMat cpy_score_grads = score_grads;
    CuVec d_weights(_handle);
    d_weights.from_eigen(softmax_scores);

    cpy_score_grads.set_dmm(d_weights, false);
    G.set_zero();
    G.set_mmT(cuX, cpy_score_grads, 1.0f / minibatch_size, 0.0f);

    // Copying second time to subtract positive class gradients
    cpy_score_grads = pos_score_grads;

    for (size_t i = 0; i < minibatch_size; ++i) {
        if (y[i] == k)
            pos_class_weights[i] = 1.0f;
        else
            pos_class_weights[i] = 0.0f;
    }

    d_weights.from_eigen(pos_class_weights);
    cpy_score_grads.set_dmm(d_weights, false);

    G.set_mmT(cuX, cpy_score_grads, -1.0f / minibatch_size, 1.0f);

    if (_hp.lambda > 0.0f)
        G.set_add(G, G, 1.0f, _hp.lambda);
}

inline void CuSmoothML3::update_G_convexified(CuMat &G, const CuMat &score_grads, const CuMat &pos_score_grads,
                                              const CuMat &cuX, const vector<uint32_t> &y,
                                              const Vec &softmax_scores,
                                              const float gamma_t, const size_t k) {
    const size_t minibatch_size = cuX._cols;
    Vec pos_class_weights(minibatch_size);
    CuMat cpy_score_grads = score_grads;
    CuVec d_weights(_handle);
    d_weights.from_eigen(softmax_scores);

    cpy_score_grads.set_dmm(d_weights, false);
    G.set_mmT(cuX, cpy_score_grads, gamma_t / minibatch_size, 1.0f - gamma_t / minibatch_size);

    // Copying second time to subtract positive class gradients
    cpy_score_grads = pos_score_grads;

    for (size_t i = 0; i < minibatch_size; ++i) {
        if (y[i] == k)
            pos_class_weights[i] = 1.0f;
        else
            pos_class_weights[i] = 0.0f;
    }

    d_weights.from_eigen(pos_class_weights);
    cpy_score_grads.set_dmm(d_weights, false);

    G.set_mmT(cuX, cpy_score_grads, -gamma_t / minibatch_size, 1.0f);
}
