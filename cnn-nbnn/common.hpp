
#pragma once

#include <iostream>

static int is_pow_of_two (unsigned int x)
{
  return ((x != 0) && ((x & (~x + 1)) == x));
}

#define __log__ std::cerr
#define __warn__ std::cerr << "WARNING: "
#define __TRACE__ std::cerr << "TRACE::" << __FILE__ << "::Line " << __LINE__ << std::endl;
#define sqr(x) ((x)*(x))

#define __TIME_INIT__ clock_t start, diff, msec;
#define __TIME_START__ start = clock();
#define __TIME_STOP__ diff = clock() - start;
//#define __TIME_REPORT__(tag) msec = diff * 1000 / CLOCKS_PER_SEC; cerr << tag << ":: Time taken " << msec/1000 << " seconds " << msec%1000 << " milliseconds" << endl;
#define __TIME_REPORT__(tag)

#define PRINT_SIZE(X, name) cerr << name << " = " << X._rows << " x " << X._cols << endl;
#define PRINT_SIZE_V(X, name) cerr << name << " = " << X._size << endl;

static float rand_gauss (void) {
  float v1,v2,s;

  do {
    v1 = 2.0 * ((float) rand()/RAND_MAX) - 1;
    v2 = 2.0 * ((float) rand()/RAND_MAX) - 1;

    s = v1*v1 + v2*v2;
  } while ( s >= 1.0 );

  if (s == 0.0)
    return 0.0;
  else
    return (v1*sqrt(-2.0 * log(s) / s));
}
