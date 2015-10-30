#include "cu_smooth_ml3_ctypes.cuh"

extern "C" CuSmoothML3* train_smooth_ml3_ctypes(TDenseDataLoaderFunc loader,
                                                const uint32_t classes, const uint32_t n, const float L, const float t0, const float p, const float lambda, const float gauss_init_std, const uint64_t pass_length,
                                                const bool record_aer, const char *model_path=NULL, const uint32_t save_model_every_t=0, const char *model_tag=NULL,
                                                TDenseDataLoaderFunc validation_loader=NULL, const bool nbnl=false) {
    CuSmoothML3 *ml3 = new CuSmoothML3(record_aer);

    CuHyperparameters hp;
    hp.n = n;
    hp.L = L;
    hp.p = p;
    hp.t0 = t0;
    hp.pass_length = pass_length;
    hp.lambda = lambda;
    hp.gauss_init_std = gauss_init_std;

    if (model_path) { // Will dump models periodically
        string s(model_path);
        ml3->set_storage_params(model_path, save_model_every_t, model_tag);
    }

    DenseExampleSet example_set(loader);
    DenseExampleSet validation_example_set(validation_loader);
    if (validation_loader) {	
	ml3->_validation_set = &validation_example_set;
    }

    ml3->_nbnl_mode = nbnl;
    ml3->train(classes, hp, example_set);

    return ml3;
}

extern "C" CuSmoothML3* resume_smooth_ml3_ctypes(TDenseDataLoaderFunc loader, const char *model_filename, const char *model_path=NULL, const uint32_t save_model_every_t=0, const char *model_tag=NULL,
						 TDenseDataLoaderFunc validation_loader=NULL, const bool nbnl=false) {
    CuSmoothML3 *ml3 = new CuSmoothML3(false);

    std::cerr << "Resuming training using model \"" << std::string(model_filename) << "\"" << std::endl;

    ml3->load(string(model_filename));

    if (model_path) { // Will dump models periodically
        string s(model_path);
        ml3->set_storage_params(model_path, save_model_every_t, model_tag);
    }

    DenseExampleSet example_set(loader);
    DenseExampleSet validation_example_set(validation_loader);
    if (validation_loader) {	
	ml3->_validation_set = &validation_example_set;
    }

    ml3->_nbnl_mode = nbnl;
    ml3->resume(ml3->_W.size(), example_set);

    return ml3;
}

extern "C" float test_smooth_ml3_ctypes(CuSmoothML3 *hand, TDenseDataLoaderFunc loader, bool on_gpu=true, bool nbnl=false) {
    DenseExampleSet example_set(loader);

    hand->_nbnl_mode = nbnl;
    return hand->test(example_set, on_gpu);
}

extern "C" CuSmoothML3* load_model_ctypes(const char *model_filename, const bool to_gpu=true) {
    CuSmoothML3 *hand = new CuSmoothML3(false);
    hand->load(string(model_filename), to_gpu);
    return hand;
}

extern "C" double get_train_time_ms_ctypes(CuSmoothML3 *hand) {
    return hand->_last_train_duration_ms;
}

extern "C" double get_test_time_ms_ctypes(CuSmoothML3 *hand) {
    return hand->_last_test_duration_ms;
}

extern "C" uint64_t get_aer_track_length(CuSmoothML3 *hand) {
    return hand->_aer_track.size();
}

extern "C" void get_aer_track(CuSmoothML3 *hand, float *aer_track) {
    uint64_t i = 0;
    for (list<float>::iterator it=hand->_aer_track.begin();
         it != hand->_aer_track.end(); ++it)
        aer_track[i++] = *it;
}

extern "C" void destroy_smooth_ml3_ctypes(CuSmoothML3 *hand) {
    delete hand;
}
