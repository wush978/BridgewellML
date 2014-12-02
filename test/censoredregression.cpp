#include <cmath>
#include <vector>
#include "FTPRL.hpp"
#include "Matrix.hpp"
#include "CensoredRegression.hpp"

double logdnorm(double x) {
  return - x * x / 2 - 0.9189385;
}

double logpnorm(double x) {
  const static double 
    b0 = 0.2316419,
    b1 = 0.319381530,
    b2 = -0.356563782,
    b3 = 1.781477937,
    b4 = -1.821255978,
    b5 = 1.330274429;
  double t = 1 / (1 + b0 * x);
  return 1 - exp(- x * x / 2) * (b1 * t + b2 * t * t + b3 * t * t * t + b4 * t * t * t * t + b5 * t * t * t * t * t) / 2.506628;
}

struct CSRMatrix : public FTPRL::Matrix<int, int> {
  
  typedef int ItorType;
  typedef int IndexType;
  
public:
  
  int *i, *p;
  double *x;
  
  CSRMatrix(int nfeature, int ninstance) 
  : FTPRL::Matrix<int, int>(nfeature, ninstance) {
  }
  
  virtual ~CSRMatrix() { }
  
  virtual ItorType getFeatureItorBegin(IndexType instance_id) const {
    return p[instance_id];
  }
  
  virtual ItorType getFeatureItorEnd(IndexType instance_id) const {
    return p[instance_id + 1];
  }
  
  virtual IndexType getFeatureId(ItorType feature_iterator) const {
    return i[feature_iterator];
  }
  
  virtual double getValue(ItorType feature_iterator) const {
    return x[feature_iterator];
  }
    

};  

bool no_skip(double y, bool is_observed) {
  return true;
}

// http://www.johndcook.com/cpp_phi.html
extern "C" {

double pnorm(double x) {
  // constants
  double a1 =  0.254829592;
  double a2 = -0.284496736;
  double a3 =  1.421413741;
  double a4 = -1.453152027;
  double a5 =  1.061405429;
  double p  =  0.3275911;

  // Save the sign of x
  int sign = 1;
  if (x < 0)
      sign = -1;
  x = fabs(x)/sqrt(2.0);

  // A&S formula 7.1.26
  double t = 1.0/(1.0 + p*x);
  double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

  return 0.5*(1.0 + sign*y);
}

double dnorm(double x) {
  return exp(- x * x / 2) / pow(2 * 3.1415926, 0.5);
}
  
}

int main() {
  
  int i[700] = {0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 
3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 
2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 
3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 
4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 
0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 
1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 
2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 
3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 
4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 
0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 
1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 
2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 
3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 
4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 
0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 
1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 2L, 3L, 4L, 0L, 1L, 
2L, 3L, 4L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 
3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 
5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 
0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 
1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 
2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 
3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 
5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 
0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 
1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 
2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 
3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 
5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 
0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 
1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 
2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L, 0L, 1L, 2L, 3L, 5L};
  int p[151] = {0L, 4L, 8L, 12L, 16L, 20L, 24L, 28L, 32L, 36L, 40L, 44L, 48L, 
52L, 56L, 60L, 64L, 68L, 72L, 76L, 80L, 84L, 88L, 92L, 96L, 100L, 
104L, 108L, 112L, 116L, 120L, 124L, 128L, 132L, 136L, 140L, 144L, 
148L, 152L, 156L, 160L, 164L, 168L, 172L, 176L, 180L, 184L, 188L, 
192L, 196L, 200L, 205L, 210L, 215L, 220L, 225L, 230L, 235L, 240L, 
245L, 250L, 255L, 260L, 265L, 270L, 275L, 280L, 285L, 290L, 295L, 
300L, 305L, 310L, 315L, 320L, 325L, 330L, 335L, 340L, 345L, 350L, 
355L, 360L, 365L, 370L, 375L, 380L, 385L, 390L, 395L, 400L, 405L, 
410L, 415L, 420L, 425L, 430L, 435L, 440L, 445L, 450L, 455L, 460L, 
465L, 470L, 475L, 480L, 485L, 490L, 495L, 500L, 505L, 510L, 515L, 
520L, 525L, 530L, 535L, 540L, 545L, 550L, 555L, 560L, 565L, 570L, 
575L, 580L, 585L, 590L, 595L, 600L, 605L, 610L, 615L, 620L, 625L, 
630L, 635L, 640L, 645L, 650L, 655L, 660L, 665L, 670L, 675L, 680L, 
685L, 690L, 695L, 700L};
  double x[750] = {1, 3.5, 1.4, 0.2, 1, 3, 1.4, 0.2, 1, 3.2, 1.3, 0.2, 1, 3.1, 
1.5, 0.2, 1, 3.6, 1.4, 0.2, 1, 3.9, 1.7, 0.4, 1, 3.4, 1.4, 0.3, 
1, 3.4, 1.5, 0.2, 1, 2.9, 1.4, 0.2, 1, 3.1, 1.5, 0.1, 1, 3.7, 
1.5, 0.2, 1, 3.4, 1.6, 0.2, 1, 3, 1.4, 0.1, 1, 3, 1.1, 0.1, 1, 
4, 1.2, 0.2, 1, 4.4, 1.5, 0.4, 1, 3.9, 1.3, 0.4, 1, 3.5, 1.4, 
0.3, 1, 3.8, 1.7, 0.3, 1, 3.8, 1.5, 0.3, 1, 3.4, 1.7, 0.2, 1, 
3.7, 1.5, 0.4, 1, 3.6, 1, 0.2, 1, 3.3, 1.7, 0.5, 1, 3.4, 1.9, 
0.2, 1, 3, 1.6, 0.2, 1, 3.4, 1.6, 0.4, 1, 3.5, 1.5, 0.2, 1, 3.4, 
1.4, 0.2, 1, 3.2, 1.6, 0.2, 1, 3.1, 1.6, 0.2, 1, 3.4, 1.5, 0.4, 
1, 4.1, 1.5, 0.1, 1, 4.2, 1.4, 0.2, 1, 3.1, 1.5, 0.2, 1, 3.2, 
1.2, 0.2, 1, 3.5, 1.3, 0.2, 1, 3.6, 1.4, 0.1, 1, 3, 1.3, 0.2, 
1, 3.4, 1.5, 0.2, 1, 3.5, 1.3, 0.3, 1, 2.3, 1.3, 0.3, 1, 3.2, 
1.3, 0.2, 1, 3.5, 1.6, 0.6, 1, 3.8, 1.9, 0.4, 1, 3, 1.4, 0.3, 
1, 3.8, 1.6, 0.2, 1, 3.2, 1.4, 0.2, 1, 3.7, 1.5, 0.2, 1, 3.3, 
1.4, 0.2, 1, 3.2, 4.7, 1.4, 1, 1, 3.2, 4.5, 1.5, 1, 1, 3.1, 4.9, 
1.5, 1, 1, 2.3, 4, 1.3, 1, 1, 2.8, 4.6, 1.5, 1, 1, 2.8, 4.5, 
1.3, 1, 1, 3.3, 4.7, 1.6, 1, 1, 2.4, 3.3, 1, 1, 1, 2.9, 4.6, 
1.3, 1, 1, 2.7, 3.9, 1.4, 1, 1, 2, 3.5, 1, 1, 1, 3, 4.2, 1.5, 
1, 1, 2.2, 4, 1, 1, 1, 2.9, 4.7, 1.4, 1, 1, 2.9, 3.6, 1.3, 1, 
1, 3.1, 4.4, 1.4, 1, 1, 3, 4.5, 1.5, 1, 1, 2.7, 4.1, 1, 1, 1, 
2.2, 4.5, 1.5, 1, 1, 2.5, 3.9, 1.1, 1, 1, 3.2, 4.8, 1.8, 1, 1, 
2.8, 4, 1.3, 1, 1, 2.5, 4.9, 1.5, 1, 1, 2.8, 4.7, 1.2, 1, 1, 
2.9, 4.3, 1.3, 1, 1, 3, 4.4, 1.4, 1, 1, 2.8, 4.8, 1.4, 1, 1, 
3, 5, 1.7, 1, 1, 2.9, 4.5, 1.5, 1, 1, 2.6, 3.5, 1, 1, 1, 2.4, 
3.8, 1.1, 1, 1, 2.4, 3.7, 1, 1, 1, 2.7, 3.9, 1.2, 1, 1, 2.7, 
5.1, 1.6, 1, 1, 3, 4.5, 1.5, 1, 1, 3.4, 4.5, 1.6, 1, 1, 3.1, 
4.7, 1.5, 1, 1, 2.3, 4.4, 1.3, 1, 1, 3, 4.1, 1.3, 1, 1, 2.5, 
4, 1.3, 1, 1, 2.6, 4.4, 1.2, 1, 1, 3, 4.6, 1.4, 1, 1, 2.6, 4, 
1.2, 1, 1, 2.3, 3.3, 1, 1, 1, 2.7, 4.2, 1.3, 1, 1, 3, 4.2, 1.2, 
1, 1, 2.9, 4.2, 1.3, 1, 1, 2.9, 4.3, 1.3, 1, 1, 2.5, 3, 1.1, 
1, 1, 2.8, 4.1, 1.3, 1, 1, 3.3, 6, 2.5, 1, 1, 2.7, 5.1, 1.9, 
1, 1, 3, 5.9, 2.1, 1, 1, 2.9, 5.6, 1.8, 1, 1, 3, 5.8, 2.2, 1, 
1, 3, 6.6, 2.1, 1, 1, 2.5, 4.5, 1.7, 1, 1, 2.9, 6.3, 1.8, 1, 
1, 2.5, 5.8, 1.8, 1, 1, 3.6, 6.1, 2.5, 1, 1, 3.2, 5.1, 2, 1, 
1, 2.7, 5.3, 1.9, 1, 1, 3, 5.5, 2.1, 1, 1, 2.5, 5, 2, 1, 1, 2.8, 
5.1, 2.4, 1, 1, 3.2, 5.3, 2.3, 1, 1, 3, 5.5, 1.8, 1, 1, 3.8, 
6.7, 2.2, 1, 1, 2.6, 6.9, 2.3, 1, 1, 2.2, 5, 1.5, 1, 1, 3.2, 
5.7, 2.3, 1, 1, 2.8, 4.9, 2, 1, 1, 2.8, 6.7, 2, 1, 1, 2.7, 4.9, 
1.8, 1, 1, 3.3, 5.7, 2.1, 1, 1, 3.2, 6, 1.8, 1, 1, 2.8, 4.8, 
1.8, 1, 1, 3, 4.9, 1.8, 1, 1, 2.8, 5.6, 2.1, 1, 1, 3, 5.8, 1.6, 
1, 1, 2.8, 6.1, 1.9, 1, 1, 3.8, 6.4, 2, 1, 1, 2.8, 5.6, 2.2, 
1, 1, 2.8, 5.1, 1.5, 1, 1, 2.6, 5.6, 1.4, 1, 1, 3, 6.1, 2.3, 
1, 1, 3.4, 5.6, 2.4, 1, 1, 3.1, 5.5, 1.8, 1, 1, 3, 4.8, 1.8, 
1, 1, 3.1, 5.4, 2.1, 1, 1, 3.1, 5.6, 2.4, 1, 1, 3.1, 5.1, 2.3, 
1, 1, 2.7, 5.1, 1.9, 1, 1, 3.2, 5.9, 2.3, 1, 1, 3.3, 5.7, 2.5, 
1, 1, 3, 5.2, 2.3, 1, 1, 2.5, 5, 1.9, 1, 1, 3, 5.2, 2, 1, 1, 
3.4, 5.4, 2.3, 1, 1, 3, 5.1, 1.8, 1};
  CSRMatrix m(6, 150);
  m.i = i;
  m.p = p;
  m.x = x;
  double y[150] = {5.1, 4.9, 4.7, 4.6, 5, 5.4, 4.6, 5, 3.4, 3.9, 5.4, 4.8, 3.8, 
4.3, 5.8, 4.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 3.6, 5.1, 4.8, 5, 
5, 5.2, 5.2, 4.7, 4.8, 4.4, 5.2, 4.5, 4.9, 5, 5.5, 4.9, 4.4, 
5.1, 5, 4.5, 4.4, 5, 4.1, 4.8, 5.1, 4.6, 5.3, 5, 7, 5.4, 6.9, 
5.5, 6.5, 4.7, 6.3, 4.9, 6.6, 5.2, 5, 4.9, 6, 6.1, 4.6, 6.7, 
5.6, 5.8, 5.2, 5.6, 5.9, 6.1, 5.3, 6.1, 6.4, 6.6, 5.8, 6.7, 6, 
5.7, 4.5, 5.5, 4.8, 6, 5.4, 6, 6.7, 6.3, 4.6, 4.5, 5.5, 6.1, 
5.8, 5, 5.6, 5.7, 5.7, 6.2, 5.1, 4.7, 6.3, 5.8, 7.1, 6.3, 6.5, 
6.6, 4.9, 7.3, 6.7, 7.2, 5.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 
7.7, 5, 6.9, 5.6, 6.7, 5.3, 6.7, 7.2, 5.2, 6.1, 6.4, 7.2, 6.4, 
7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6, 5.9, 6.7, 6.9, 5.8, 6.8, 
5.7, 6.7, 5.3, 6.5, 6.2, 5.9};
  bool is_observed[150] = {true, true, true, true, true, true, true, true, false, false, 
true, true, false, true, true, false, true, true, true, true, 
true, true, false, true, true, true, true, true, true, true, 
true, false, true, false, true, true, true, true, true, true, 
true, true, true, true, false, true, true, true, true, true, 
true, false, true, true, true, false, true, true, true, true, 
true, false, true, true, false, true, true, true, false, true, 
true, true, false, true, true, true, false, true, true, true, 
false, true, false, true, true, true, true, true, false, false, 
true, true, true, true, true, true, true, true, true, false, 
true, true, true, true, true, false, true, true, true, true, 
false, true, true, true, true, true, true, true, true, false, 
true, true, false, false, true, true, false, true, true, true, 
false, true, true, true, true, true, true, true, true, false, 
true, true, true, true, false, true, false, true, true, true};

  FTPRL::FTPRL ftprl(0.1, 1, 0.1, 0.1);
  FTPRL::CensoredRegression<int> lr(&ftprl, 5);
  lr.update<int, double, bool>(&m, y, is_observed, no_skip);
  
  std::vector<double> rp(150, 0.0);
  lr.predict<int>(&m, &rp[0]);
  
  return 0;
}

