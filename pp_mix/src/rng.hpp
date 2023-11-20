#ifndef RNG_HPP
#define RNG_HPP

#include <stan/math/prim.hpp>
#include <random>

class Rng {
 public:
  //! Returns (and creates if nonexistent) the singleton of this class
  static Rng &Instance() {
    static Rng s;
    return s;
  }

  //! Returns a reference to the underlying RNG object
  std::mt19937 &get() { return mt; }

  //! Sets the RNG seed
  void seed(const int seed_val) { mt.seed(seed_val); }

 private:
  Rng(const int seed_val = 20201103) { mt.seed(seed_val); }
  ~Rng() {}
  Rng(Rng const &) = delete;
  Rng &operator=(Rng const &) = delete;

  //! C++ standard library RNG object
  std::mt19937 mt;
};

#endif
