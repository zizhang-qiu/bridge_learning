//
// Created by qzz on 2023/11/12.
//

#include "pareto_front.h"
#include "bridge_lib/utils.h"
void ProductTest() {
  const OutcomeVector vec1{{0, 1, 1}, {true, true, true}};
  const OutcomeVector vec2{{1, 1, 0}, {true, true, true}};
  const ParetoFront pf1{{vec1, vec2}};

  const OutcomeVector vec3{{1, 1, 0}, {true, true, true}};
  const OutcomeVector vec4{{1, 0, 1}, {true, true, true}};
  const ParetoFront pf2{{vec3, vec4}};

  const ParetoFront product = pf1 * pf2;
  const OutcomeVector vec5{{0, 0, 1}, {true, true, true}};
  const OutcomeVector vec6{{1, 1, 0}, {true, true, true}};
  const ParetoFront expected_product{{vec5, vec6}};
  REQUIRE(product==expected_product);
}

int main() {
  ProductTest();
  return 0;
}