//
// Created by qzz on 2023/11/12.
//

#include "playcc/pareto_front.h"
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
  REQUIRE(product == expected_product);
}

void SerializationTest() {
  const OutcomeVector vec1{{0, 1, 1}, {true, true, true},
                           {ble::BridgeMove::Type::kPlay, ble::kClubsSuit, 2, ble::kInvalidDenomination, -1,
                            ble::kNotOtherCall}};
  const OutcomeVector vec2{{1, 1, 0}, {true, true, true}, {}};
  const ParetoFront pf1{{vec1, vec2}};
  const std::string serialized = pf1.Serialize();
//  std::cout << serialized << std::endl;
  const std::string expected = "game status\n0\n1\n1\npossible worlds\n1\n1\n1\nmove\nC4\n\ngame status\n1\n1\n0\npossible worlds\n1\n1\n1\nmove\nII\n\n";
  REQUIRE_EQ(serialized, expected);
  REQUIRE_EQ(ParetoFront::Deserialize(serialized), pf1);

  // Empty front.
  const ParetoFront pf2{};
  const std::string serialized2 = pf2.Serialize();
  const std::string expected2;
  REQUIRE_EQ(serialized2, expected2);

  REQUIRE_EQ(ParetoFront::Deserialize(serialized2), pf2);
}

int main() {
  ProductTest();
  SerializationTest();
  return 0;
}