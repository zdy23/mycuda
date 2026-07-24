#include <cstdint>
#include <iostream>
#include <sys/types.h>

class solution {
public:
  uint32_t reverseBits(uint32_t n) {
    if (n == 0)
      return 0;
    uint32_t res = 0;
    for (int i = 0; i < 32; i++) {
      uint32_t temp = n % 2;
      n = n >> 1;
      res = (res << 1) + temp;
    }
    return res;
  }
};

int main() {
  solution s;
  uint32_t num = 43261596;
  std::cout << s.reverseBits(num) << std::endl;
  return 0;
}