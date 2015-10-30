
#pragma once

#include <memory>
#include "CuSmoothML3.cuh"


struct DenseDataBatch {
    float *data;
    int cols;
    int rows;

    uint32_t *y;
    uint32_t *raw_tags;

    bool finished;

    DenseDataBatch()
	: data(0), cols(0), rows(0), y(0), raw_tags(0), finished(false)
    {}

    MappedMat convert(vector<uint32_t> &labels, vector<uint32_t> &tags) {
        MappedMat mat(data, rows, cols);

        // cerr << "First column of dense mapped mat:" << endl
        //      << mat.col(0) << endl;

        labels.resize(cols);
        for (uint32_t i = 0; i < cols; ++i) {
            labels[i] = y[i];
        }

	if (raw_tags) { // tags are optional
	    tags.resize(cols);
	    for (uint32_t i = 0; i < cols; ++i) {
		tags[i] = raw_tags[i];
	    }
	}

        return mat;
    }
};

typedef void TDenseDataLoaderFunc(DenseDataBatch*);

template<class TBatch, class TLoader>
class OnlineExampleSet : public MinibatchExampleSet {
    TLoader *_loader;
    bool _finished;
    Mat _X;
    vector<uint32_t> _labels;
    vector<uint32_t> _tags;

public:
    OnlineExampleSet(TLoader *loader) : _loader(loader), _finished(false) {}

    virtual bool get_minibatch(Minibatch &mb) {
        if (_finished)
            return false;

        TBatch b;
        //__log__ << "Trying to load new batch..." << endl;

        _loader(&b);
        _finished = b.finished;

        if (!_finished) {
            _X = b.convert(_labels, _tags);
            //__log__ << "Done." << endl;
        }
        else {
            //__log__ << "No batches left." << endl;
            return false;
        }
        mb.X = &_X;
        mb.y = &_labels;
	mb.tags = &_tags;

        return true;
    }

    virtual void reset() {
        _finished = false;
    }
};

typedef OnlineExampleSet<DenseDataBatch, TDenseDataLoaderFunc>  DenseExampleSet;

#ifdef DEBUG

template<class TBatch, class TLoader>
class OnlineExampleSet_DEBUG : public MinibatchExampleSet {
    TLoader *_loader;
    uint32_t _example_iter;
    bool _finished;
    Mat _X;
    vector<uint32_t> _labels;
    vector<uint32_t> _tmp_labels;
    Mat _tmp_X;

public:
    OnlineExampleSet_DEBUG(TLoader *loader) : _loader(loader), _example_iter(0), _finished(false) {}

    virtual bool get_minibatch(Minibatch &mb) {
        if (_finished)
            return false;

        if ( _example_iter == _X.cols() ) {
            TBatch b;
            __log__ << "Trying to load new batch..." << endl;

            _loader(&b);
            _finished = b.finished;

            if (!_finished) {
                _X = b.convert(_labels);
                _example_iter = 0;
                __log__ << "Done." << endl;
            }
            else {
                __log__ << "No batches left." << endl;
                return false;
            }
        }

        _tmp_X = _X.col(_example_iter);
        _tmp_labels.clear();
        _tmp_labels.push_back(_labels[_example_iter]);

        mb.X = &_tmp_X;
        mb.y = &_tmp_labels;

        _example_iter++;
        return true;
    }
};

typedef OnlineExampleSet_DEBUG<DenseDataBatch, TDenseDataLoaderFunc>  DenseExampleSet_DEBUG;

#endif
