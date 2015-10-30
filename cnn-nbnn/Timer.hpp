
#include <assert.h>
#include <time.h>
#include <iostream>
#include <sstream>

using namespace std;

struct Timer {
    unsigned long _total;
    unsigned long _start;
    bool _paused;

    Timer()
        : _total(0), _start(0), _paused(true) {}

    inline void reset() {
        _total = 0;
    }

    inline void start() {
        assert(_paused);
        _paused = false;

        _start = clock();
    }

    inline void stop() {
        _total = clock() - _start;
        _start = 0;
    }

    inline void pause() {
        assert(!_paused);
        _paused = true;

        _total += (clock() - _start);
        _start = 0;
    }

    inline float get_ms() {
        return _total / (CLOCKS_PER_SEC/1000);
    }

    inline string get_string() {
        size_t msec = get_ms();
        stringstream ss;
        ss << msec/1000 << " seconds " << msec%1000 << " milliseconds.";
        return ss.str();
    }

    void save(ofstream &out) {
        out.write((char*) (&_total), sizeof(_total));
        out.write((char*) (&_start), sizeof(_start));
    }

    void load(ifstream &out) {
        out.read((char*) (&_total), sizeof(_total));
        out.read((char*) (&_start), sizeof(_start));
    }
};
