//#define NOMINMAX
//#include "bridge_lib/canonical_encoder.h"
//#include "bridge_lib/bridge_observation.h"
//#include "bridge_lib/example_cards_ddts.h"
//#include "rela/batch_runner.h"
//#include "rlcc/bridge_env.h"
#include<stdio.h>
#include<string.h>
#include <sstream>
#include <iostream>
#include "playcc/pareto_front.h"
int main() {
  auto f = ParetoFront();
  f.Insert({{1, 0, 0}, {true, true, true}});
  f.Insert({{0, 1, 1}, {true, true, true}});
  std::cout << f.ToString() << std::endl;
}