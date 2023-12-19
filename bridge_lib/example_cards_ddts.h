//
// Created by qzz on 2023/9/19.
//

#ifndef BRIDGE_LIB_BRIDGE_LIB_EXAMPLE_CARDS_DDTS_H_
#define BRIDGE_LIB_BRIDGE_LIB_EXAMPLE_CARDS_DDTS_H_
#include <vector>

#include "bridge_utils.h"
namespace bridge_learning_env {
// Some cards and ddts for debug and test.

const std::vector<std::vector<Action>> example_deals = {
    {47, 39, 43, 51, 35, 27, 3, 19, 31, 15, 46, 26, 23, 30, 38, 2, 11, 41, 22, 49, 7, 17, 18, 45, 50, 44, 14, 21, 42,
     40, 33, 13, 34, 36, 29, 9, 10, 28, 25, 5, 6, 12, 32, 1, 37, 4, 16, 48, 20, 0, 8, 24},
    {51, 39, 27, 7, 47, 31, 23, 3, 43, 19, 15, 38, 35, 50, 11, 10, 42, 46, 26, 6, 34, 30, 18, 37, 22, 2, 14, 33, 49, 21,
     45, 29, 25, 9, 41, 1, 44, 5, 17, 32, 40, 24, 13, 28, 36, 16, 48, 20, 4, 12, 0, 8},
    {43, 47, 31, 51, 15, 50, 27, 39, 11, 46, 23, 35, 34, 45, 19, 3, 26, 37, 7, 42, 22, 21, 49, 38, 14, 5, 25, 30, 6, 44,
     9, 18, 2, 36, 1, 10, 33, 32, 40, 41, 29, 20, 24, 48, 17, 8, 16, 28, 13, 0, 12, 4},
    {51, 31, 47, 43, 39, 11, 27, 35, 7, 34, 15, 23, 3, 22, 50, 19, 38, 18, 42, 46, 26, 49, 30, 14, 2, 45, 25, 10, 33,
     41, 21, 6, 13, 5, 36, 37, 9, 1, 20, 29, 48, 44, 16, 17, 32, 28, 8, 40, 4, 12, 0, 24},
    {47, 43, 51, 19, 31, 39, 23, 11, 27, 35, 7, 46, 15, 3, 22, 38, 45, 50, 14, 30, 33, 42, 6, 26, 29, 34, 2, 10, 5, 18,
     41, 49, 1, 17, 25, 37, 32, 13, 36, 21, 20, 48, 28, 9, 8, 24, 12, 44, 0, 16, 4, 40},
    {31, 43, 51, 39, 15, 35, 47, 7, 42, 27, 19, 34, 30, 23, 50, 2, 18, 11, 46, 41, 14, 3, 38, 29, 10, 6, 26, 25, 49, 45,
     22, 9, 21, 37, 33, 48, 1, 17, 5, 40, 36, 13, 28, 32, 24, 44, 8, 16, 20, 0, 4, 12},
    {51, 19, 35, 47, 43, 15, 46, 31, 39, 42, 38, 27, 23, 14, 34, 7, 11, 6, 22, 30, 3, 2, 29, 26, 50, 49, 25, 18, 45, 41,
     5, 10, 21, 17, 48, 37, 1, 13, 44, 33, 12, 9, 40, 32, 4, 28, 36, 24, 0, 8, 20, 16},
    {43, 47, 27, 51, 15, 7, 19, 39, 11, 3, 22, 35, 50, 46, 14, 31, 34, 38, 10, 23, 30, 26, 6, 42, 49, 41, 37, 18, 45,
     25, 13, 2, 29, 17, 1, 33, 44, 9, 48, 21, 28, 40, 36, 5, 4, 16, 32, 20, 0, 12, 24, 8},
    {39, 42, 43, 51, 35, 38, 23, 47, 31, 22, 11, 27, 19, 10, 7, 3, 15, 6, 30, 46, 50, 33, 18, 41, 34, 25, 14, 37, 26,
     21, 2, 29, 49, 17, 45, 9, 1, 13, 48, 44, 40, 5, 32, 36, 24, 12, 20, 28, 8, 0, 16, 4},
    {43, 47, 15, 51, 46, 31, 11, 39, 42, 23, 7, 35, 34, 19, 3, 27, 22, 38, 18, 50, 6, 26, 10, 30, 2, 37, 25, 14, 29, 17,
     13, 49, 21, 1, 5, 45, 32, 48, 44, 41, 20, 36, 40, 33, 16, 28, 8, 9, 12, 0, 4, 24},
    {39, 47, 35, 51, 31, 43, 15, 23, 27, 11, 46, 3, 19, 7, 42, 34, 50, 22, 38, 14, 2, 18, 30, 10, 29, 6, 26, 49, 9, 33,
     37, 45, 5, 21, 17, 41, 1, 13, 48, 25, 44, 28, 32, 40, 4, 24, 20, 36, 0, 16, 8, 12},
    {39, 47, 27, 51, 23, 31, 19, 43, 11, 15, 38, 35, 34, 7, 6, 46, 18, 3, 45, 42, 49, 50, 37, 30, 41, 22, 25, 26, 33, 2,
     1, 14, 21, 17, 48, 10, 44, 13, 32, 29, 36, 5, 8, 9, 28, 40, 4, 16, 24, 20, 0, 12},
    {31, 43, 51, 47, 3, 27, 35, 39, 46, 42, 7, 23, 14, 34, 50, 19, 10, 30, 38, 15, 2, 26, 22, 11, 21, 6, 18, 41, 13, 45,
     49, 33, 48, 37, 25, 17, 32, 29, 9, 5, 28, 36, 40, 1, 12, 24, 20, 44, 0, 16, 8, 4},
    {51, 23, 15, 47, 43, 19, 11, 35, 39, 7, 46, 50, 31, 3, 30, 38, 27, 42, 26, 10, 22, 34, 45, 6, 18, 14, 21, 29, 49, 2,
     13, 25, 41, 37, 40, 9, 17, 33, 32, 48, 44, 5, 20, 24, 28, 1, 16, 12, 0, 36, 4, 8},
    {51, 23, 47, 39, 31, 15, 43, 11, 27, 7, 35, 34, 19, 3, 30, 14, 38, 50, 22, 49, 26, 46, 18, 41, 10, 42, 37, 33, 2, 6,
     21, 17, 13, 45, 1, 48, 9, 29, 40, 32, 5, 25, 8, 24, 28, 44, 4, 20, 16, 36, 0, 12},
    {51, 47, 39, 15, 35, 43, 23, 42, 50, 31, 19, 26, 34, 27, 7, 10, 30, 11, 3, 6, 22, 38, 46, 45, 18, 49, 2, 33, 14, 41,
     37, 17, 36, 25, 29, 9, 24, 32, 21, 5, 16, 28, 13, 1, 8, 20, 48, 44, 4, 0, 40, 12},
    {47, 39, 51, 35, 27, 15, 43, 31, 46, 3, 19, 23, 26, 38, 11, 7, 22, 30, 50, 34, 6, 10, 42, 14, 49, 2, 18, 37, 29, 33,
     45, 17, 25, 5, 41, 13, 28, 44, 21, 9, 16, 36, 40, 1, 8, 24, 12, 48, 0, 20, 4, 32},
    {23, 47, 39, 51, 15, 3, 35, 43, 30, 38, 11, 31, 10, 34, 46, 27, 6, 49, 42, 19, 45, 37, 18, 7, 41, 29, 2, 50, 33, 1,
     21, 26, 9, 44, 17, 22, 5, 40, 13, 14, 24, 32, 48, 25, 20, 8, 36, 12, 16, 0, 28, 4},
    {35, 39, 47, 51, 50, 31, 27, 43, 38, 23, 11, 19, 26, 7, 3, 15, 22, 34, 2, 46, 10, 30, 45, 42, 6, 14, 41, 18, 37, 21,
     29, 49, 25, 48, 17, 33, 13, 44, 9, 1, 5, 20, 32, 40, 36, 16, 28, 24, 12, 4, 0, 8},
    {47, 27, 51, 39, 19, 11, 43, 35, 46, 50, 31, 23, 38, 42, 15, 3, 30, 18, 7, 2, 26, 14, 34, 45, 22, 10, 25, 13, 41, 6,
     21, 9, 37, 49, 5, 44, 33, 1, 28, 40, 29, 48, 16, 36, 17, 12, 4, 32, 24, 8, 0, 20},
    {43, 51, 31, 47, 34, 39, 23, 35, 30, 27, 19, 15, 26, 11, 7, 50, 49, 14, 3, 46, 45, 6, 42, 38, 33, 2, 22, 10, 29, 41,
     18, 25, 13, 37, 9, 21, 5, 17, 1, 48, 28, 40, 32, 44, 20, 36, 16, 24, 0, 8, 12, 4},
    {51, 47, 35, 39, 43, 27, 15, 31, 23, 18, 3, 42, 19, 6, 46, 34, 11, 2, 10, 26, 7, 49, 29, 22, 50, 41, 21, 45, 38, 25,
     13, 37, 30, 17, 9, 1, 14, 44, 5, 40, 33, 12, 32, 36, 48, 4, 24, 28, 8, 0, 20, 16},
    {31, 47, 51, 43, 27, 39, 11, 35, 23, 19, 3, 46, 7, 15, 30, 42, 38, 50, 18, 34, 22, 26, 10, 2, 14, 45, 49, 41, 6, 25,
     37, 33, 17, 21, 13, 29, 48, 9, 40, 5, 24, 28, 36, 1, 20, 16, 32, 44, 0, 8, 4, 12},
    {47, 43, 27, 51, 35, 39, 7, 3, 31, 23, 50, 34, 19, 15, 46, 18, 11, 42, 30, 14, 10, 38, 22, 6, 37, 26, 49, 45, 33, 2,
     41, 1, 25, 21, 29, 44, 17, 13, 9, 40, 20, 5, 28, 36, 12, 48, 8, 24, 0, 32, 4, 16},
    {39, 19, 51, 35, 46, 11, 47, 23, 38, 50, 43, 7, 34, 18, 31, 42, 10, 14, 27, 30, 49, 29, 15, 26, 41, 21, 3, 22, 37,
     17, 2, 6, 25, 13, 45, 44, 48, 1, 33, 32, 28, 40, 9, 20, 16, 24, 5, 12, 4, 8, 36, 0},
    {51, 39, 47, 43, 31, 35, 23, 27, 15, 11, 7, 19, 50, 34, 3, 18, 46, 6, 38, 10, 42, 2, 30, 49, 14, 41, 26, 45, 33, 37,
     22, 9, 21, 25, 29, 40, 5, 13, 17, 36, 24, 44, 1, 28, 16, 32, 48, 20, 4, 8, 0, 12},
    {51, 35, 27, 31, 47, 3, 19, 23, 43, 38, 11, 7, 39, 22, 42, 50, 15, 18, 30, 46, 34, 14, 26, 37, 10, 49, 45, 25, 6,
     21, 29, 17, 2, 13, 48, 9, 41, 36, 44, 1, 33, 24, 40, 32, 5, 16, 28, 20, 0, 8, 12, 4},
    {47, 23, 26, 51, 43, 46, 18, 39, 35, 34, 10, 19, 31, 49, 2, 15, 27, 33, 45, 11, 3, 21, 37, 7, 50, 17, 25, 42, 30,
     13, 1, 38, 14, 5, 40, 22, 29, 44, 28, 6, 9, 36, 24, 41, 32, 20, 8, 48, 12, 16, 4, 0},
    {27, 31, 47, 51, 15, 19, 43, 39, 7, 50, 11, 35, 42, 22, 3, 23, 30, 14, 38, 46, 2, 10, 34, 26, 41, 6, 49, 18, 37, 25,
     45, 29, 9, 13, 33, 5, 36, 1, 21, 28, 24, 44, 17, 16, 20, 40, 48, 8, 12, 32, 0, 4},
    {47, 43, 51, 23, 27, 15, 39, 19, 50, 7, 35, 11, 26, 42, 31, 3, 10, 38, 34, 46, 6, 30, 18, 45, 2, 22, 14, 41, 13, 49,
     37, 29, 44, 21, 33, 25, 36, 5, 1, 17, 28, 48, 24, 9, 20, 40, 16, 12, 4, 32, 8, 0},
    {15, 51, 39, 31, 50, 47, 35, 27, 34, 43, 23, 3, 26, 19, 7, 46, 22, 11, 42, 38, 6, 49, 30, 18, 2, 41, 45, 14, 9, 21,
     37, 10, 44, 1, 33, 29, 40, 36, 17, 25, 28, 24, 32, 13, 8, 16, 20, 5, 0, 12, 4, 48},
    {43, 23, 51, 47, 31, 11, 27, 39, 15, 3, 19, 35, 7, 50, 30, 38, 42, 46, 2, 22, 34, 26, 25, 18, 37, 14, 21, 10, 33,
     45, 5, 6, 17, 41, 1, 49, 9, 29, 36, 13, 48, 32, 12, 44, 40, 28, 8, 24, 16, 20, 4, 0},
    {51, 39, 47, 11, 42, 35, 43, 50, 34, 31, 27, 38, 30, 19, 23, 18, 26, 7, 15, 14, 2, 46, 3, 41, 25, 22, 10, 29, 13,
     37, 6, 21, 1, 44, 49, 9, 48, 40, 45, 5, 36, 32, 33, 28, 24, 20, 17, 8, 4, 12, 16, 0},
    {43, 47, 51, 35, 39, 15, 19, 31, 23, 3, 7, 27, 42, 2, 34, 11, 38, 49, 22, 50, 30, 21, 18, 46, 14, 5, 45, 26, 10, 1,
     37, 6, 41, 48, 29, 33, 17, 44, 9, 25, 13, 40, 24, 20, 28, 36, 8, 16, 4, 32, 0, 12},
    {43, 39, 47, 51, 35, 46, 31, 23, 27, 30, 7, 15, 19, 22, 50, 3, 11, 14, 38, 42, 2, 6, 34, 26, 45, 33, 49, 18, 25, 21,
     41, 10, 13, 9, 5, 37, 40, 1, 24, 29, 36, 48, 16, 17, 28, 44, 12, 20, 0, 32, 8, 4},
    {47, 35, 51, 39, 43, 15, 23, 31, 11, 50, 42, 27, 3, 30, 34, 19, 38, 22, 14, 7, 18, 10, 6, 46, 13, 37, 45, 26, 44, 1,
     41, 2, 36, 48, 33, 49, 32, 40, 29, 17, 20, 28, 25, 9, 16, 24, 21, 5, 12, 4, 0, 8},
    {31, 23, 47, 51, 27, 50, 43, 15, 19, 42, 39, 7, 34, 38, 35, 46, 22, 41, 11, 30, 10, 21, 3, 18, 6, 17, 26, 14, 45,
     48, 49, 2, 33, 44, 29, 37, 13, 32, 9, 25, 5, 24, 36, 1, 4, 12, 28, 40, 0, 8, 16, 20},
    {51, 43, 19, 39, 47, 35, 3, 23, 27, 31, 50, 11, 15, 7, 46, 26, 30, 42, 34, 18, 10, 38, 22, 2, 45, 49, 14, 41, 37,
     33, 6, 21, 13, 17, 29, 48, 1, 9, 25, 40, 44, 36, 5, 16, 28, 32, 20, 8, 12, 24, 4, 0},
    {39, 43, 51, 47, 35, 27, 42, 31, 7, 19, 30, 23, 50, 15, 26, 11, 46, 3, 22, 38, 10, 34, 18, 49, 6, 2, 14, 37, 25, 29,
     45, 33, 40, 17, 41, 21, 36, 5, 44, 13, 28, 1, 32, 9, 24, 20, 8, 48, 4, 16, 0, 12},
    {43, 51, 19, 39, 35, 47, 3, 31, 38, 27, 46, 11, 26, 23, 34, 50, 14, 15, 10, 30, 2, 7, 41, 22, 45, 42, 29, 18, 9, 33,
     25, 6, 48, 21, 17, 49, 36, 13, 1, 37, 24, 28, 44, 5, 16, 4, 20, 40, 12, 0, 8, 32},
    {51, 27, 47, 39, 15, 19, 43, 35, 3, 11, 31, 7, 50, 42, 23, 46, 34, 26, 14, 38, 30, 18, 45, 2, 22, 10, 37, 41, 6, 49,
     25, 33, 29, 17, 1, 21, 13, 5, 48, 9, 36, 40, 32, 44, 16, 12, 28, 24, 0, 8, 20, 4},
    {31, 39, 51, 47, 27, 23, 43, 35, 11, 15, 19, 46, 3, 7, 50, 38, 30, 42, 34, 49, 18, 14, 26, 37, 10, 2, 22, 25, 33,
     41, 6, 13, 21, 29, 45, 44, 9, 17, 5, 32, 1, 48, 36, 28, 40, 20, 24, 12, 16, 0, 4, 8},
    {47, 35, 51, 31, 39, 27, 43, 23, 3, 15, 19, 46, 50, 11, 18, 34, 42, 7, 6, 14, 38, 41, 2, 45, 30, 29, 49, 33, 26, 9,
     25, 21, 22, 5, 28, 17, 10, 1, 16, 13, 37, 44, 8, 48, 36, 24, 4, 40, 20, 12, 0, 32},
    {27, 51, 35, 47, 23, 43, 31, 11, 19, 39, 18, 3, 15, 50, 10, 42, 7, 46, 49, 30, 34, 38, 29, 22, 6, 26, 9, 14, 2, 33,
     5, 37, 45, 17, 1, 25, 41, 44, 28, 48, 21, 40, 20, 36, 13, 24, 8, 16, 32, 0, 4, 12},
    {39, 35, 51, 43, 38, 23, 47, 31, 26, 15, 42, 27, 14, 11, 34, 19, 37, 3, 30, 7, 17, 22, 10, 50, 13, 45, 6, 46, 9, 21,
     2, 18, 48, 1, 49, 33, 44, 28, 41, 29, 20, 12, 40, 25, 16, 8, 36, 5, 4, 0, 24, 32},
    {47, 51, 39, 43, 31, 27, 19, 35, 23, 15, 11, 7, 42, 38, 3, 10, 34, 30, 50, 6, 14, 22, 46, 41, 49, 45, 26, 21, 37,
     33, 18, 9, 25, 29, 2, 5, 13, 17, 44, 40, 48, 1, 36, 28, 16, 32, 20, 12, 8, 24, 4, 0},
    {51, 27, 39, 47, 43, 23, 15, 31, 35, 19, 38, 18, 11, 7, 26, 2, 46, 3, 10, 49, 34, 50, 6, 41, 30, 42, 37, 29, 22, 33,
     17, 5, 14, 25, 13, 1, 45, 21, 9, 48, 28, 40, 44, 24, 20, 16, 36, 4, 12, 8, 32, 0},
    {51, 39, 43, 47, 31, 35, 11, 23, 27, 22, 7, 19, 15, 18, 10, 50, 3, 6, 41, 46, 42, 21, 33, 34, 38, 9, 29, 30, 26, 36,
     25, 14, 49, 24, 1, 2, 37, 16, 32, 45, 13, 12, 28, 17, 5, 4, 20, 48, 44, 0, 8, 40},
    {47, 39, 35, 51, 43, 19, 27, 7, 31, 3, 23, 14, 15, 50, 11, 10, 22, 38, 46, 2, 6, 34, 42, 29, 49, 30, 26, 17, 41, 37,
     18, 13, 48, 33, 45, 36, 40, 21, 25, 32, 24, 9, 1, 28, 20, 5, 12, 16, 0, 44, 4, 8},
    {47, 35, 23, 51, 39, 31, 11, 43, 3, 19, 7, 27, 42, 15, 46, 6, 30, 50, 38, 2, 18, 22, 34, 41, 10, 33, 26, 17, 37, 5,
     14, 9, 29, 1, 49, 48, 25, 24, 45, 40, 13, 20, 21, 36, 44, 4, 32, 28, 12, 0, 16, 8},
    {47, 39, 15, 51, 35, 3, 11, 43, 23, 46, 7, 31, 14, 42, 38, 27, 49, 22, 34, 19, 45, 6, 33, 50, 41, 37, 29, 30, 17,
     21, 5, 26, 13, 9, 1, 18, 32, 40, 48, 10, 8, 24, 36, 2, 4, 16, 28, 25, 0, 12, 20, 44},
    {47, 43, 51, 31, 42, 39, 35, 27, 18, 23, 11, 19, 10, 3, 38, 15, 6, 26, 34, 7, 49, 22, 2, 50, 45, 41, 5, 46, 17, 25,
     48, 30, 9, 21, 36, 14, 44, 13, 24, 37, 40, 1, 20, 33, 16, 32, 8, 29, 12, 28, 4, 0},
    {51, 47, 43, 31, 35, 39, 15, 27, 19, 23, 50, 11, 7, 3, 38, 46, 14, 30, 34, 42, 45, 22, 26, 18, 41, 2, 6, 10, 37, 49,
     9, 17, 25, 33, 5, 1, 21, 29, 48, 16, 40, 13, 32, 12, 28, 44, 24, 8, 20, 36, 4, 0},
    {51, 43, 47, 27, 35, 39, 19, 15, 46, 31, 34, 11, 42, 23, 22, 3, 26, 7, 2, 38, 18, 50, 45, 30, 14, 10, 37, 6, 49, 29,
     13, 41, 33, 17, 5, 25, 21, 48, 36, 9, 44, 24, 20, 1, 28, 16, 4, 40, 8, 12, 0, 32},
    {35, 39, 51, 19, 23, 27, 47, 7, 15, 46, 43, 3, 11, 14, 31, 38, 42, 10, 50, 26, 30, 45, 34, 18, 2, 29, 22, 6, 37, 25,
     17, 49, 1, 21, 13, 41, 44, 5, 9, 33, 40, 48, 20, 32, 36, 24, 12, 28, 0, 16, 4, 8},
    {39, 51, 11, 43, 31, 47, 7, 35, 27, 26, 3, 23, 50, 22, 46, 19, 42, 10, 38, 15, 14, 17, 18, 34, 2, 13, 6, 30, 49, 40,
     33, 41, 45, 32, 9, 37, 21, 28, 5, 29, 44, 16, 48, 25, 36, 8, 12, 1, 20, 4, 0, 24},
    {51, 35, 47, 43, 39, 31, 15, 23, 19, 27, 7, 26, 46, 11, 3, 6, 42, 50, 34, 49, 38, 30, 33, 29, 22, 14, 13, 25, 18, 2,
     9, 17, 10, 41, 44, 5, 45, 37, 36, 1, 21, 48, 12, 40, 32, 28, 8, 20, 16, 24, 0, 4},
    {51, 35, 27, 47, 43, 7, 15, 39, 31, 50, 3, 23, 19, 46, 22, 42, 11, 38, 18, 14, 30, 34, 2, 10, 26, 25, 41, 6, 49, 13,
     17, 45, 33, 9, 44, 37, 21, 1, 40, 29, 36, 48, 24, 5, 20, 4, 16, 32, 8, 0, 12, 28},
    {23, 43, 51, 27, 19, 39, 47, 7, 38, 15, 35, 46, 14, 11, 31, 34, 10, 3, 50, 18, 49, 42, 30, 2, 45, 22, 26, 41, 33,
     37, 6, 25, 29, 13, 21, 17, 40, 5, 48, 9, 32, 1, 16, 36, 28, 44, 12, 8, 24, 20, 4, 0},
    {51, 39, 47, 43, 35, 7, 31, 23, 27, 42, 15, 46, 19, 22, 11, 38, 18, 45, 3, 34, 2, 25, 50, 30, 17, 1, 26, 14, 13, 44,
     10, 6, 5, 40, 29, 49, 32, 36, 21, 41, 24, 28, 9, 37, 20, 16, 48, 33, 8, 4, 12, 0},
    {51, 39, 47, 27, 35, 19, 43, 34, 23, 15, 31, 22, 50, 7, 11, 18, 38, 46, 3, 6, 26, 30, 42, 2, 14, 10, 49, 33, 25, 37,
     45, 29, 17, 21, 41, 1, 5, 13, 9, 48, 44, 36, 32, 40, 20, 24, 28, 16, 4, 8, 0, 12},
    {43, 51, 19, 47, 35, 31, 30, 39, 27, 23, 18, 15, 46, 3, 14, 11, 34, 38, 10, 7, 22, 26, 41, 50, 2, 37, 13, 42, 49,
     33, 9, 6, 45, 44, 1, 25, 29, 36, 48, 21, 32, 24, 40, 17, 28, 16, 20, 5, 12, 4, 0, 8},
    {39, 51, 47, 43, 15, 35, 50, 31, 11, 23, 38, 27, 7, 19, 26, 46, 34, 3, 18, 42, 33, 22, 10, 30, 29, 14, 6, 45, 17,
     49, 2, 41, 5, 9, 13, 37, 1, 48, 40, 25, 32, 44, 24, 21, 16, 36, 20, 28, 0, 4, 12, 8},
    {39, 51, 43, 47, 31, 27, 35, 11, 19, 38, 23, 3, 15, 26, 7, 22, 46, 18, 50, 10, 42, 41, 34, 6, 30, 29, 14, 49, 45,
     13, 2, 17, 25, 40, 37, 9, 48, 36, 33, 5, 24, 28, 21, 1, 8, 16, 32, 44, 4, 12, 20, 0},
    {31, 51, 7, 43, 23, 47, 50, 27, 19, 39, 10, 42, 15, 35, 49, 34, 11, 38, 41, 22, 3, 26, 33, 18, 46, 37, 25, 2, 30,
     17, 21, 45, 14, 9, 13, 5, 6, 48, 40, 1, 29, 44, 36, 16, 24, 32, 28, 12, 20, 0, 8, 4},
    {11, 51, 47, 35, 50, 39, 43, 30, 42, 27, 31, 26, 6, 19, 23, 14, 2, 15, 7, 49, 41, 46, 3, 45, 29, 22, 38, 33, 21, 10,
     34, 13, 9, 25, 18, 36, 5, 17, 37, 32, 1, 24, 48, 28, 44, 16, 40, 4, 20, 12, 8, 0},
    {51, 47, 31, 39, 43, 35, 27, 19, 7, 15, 23, 3, 34, 50, 11, 30, 26, 46, 42, 18, 6, 2, 38, 14, 45, 49, 22, 37, 33, 21,
     10, 29, 40, 9, 41, 17, 32, 36, 25, 13, 24, 28, 1, 5, 4, 20, 48, 44, 0, 12, 8, 16},
    {23, 47, 43, 51, 19, 35, 31, 39, 34, 15, 27, 3, 18, 11, 7, 26, 10, 30, 50, 22, 2, 6, 46, 49, 9, 29, 42, 45, 48, 21,
     38, 37, 36, 13, 14, 33, 32, 5, 41, 17, 24, 40, 25, 1, 12, 16, 44, 20, 0, 4, 28, 8},
    {51, 47, 31, 35, 39, 43, 27, 3, 15, 19, 23, 18, 7, 11, 46, 14, 38, 50, 42, 6, 30, 34, 26, 2, 45, 22, 33, 37, 41, 10,
     13, 21, 29, 49, 9, 1, 17, 25, 5, 48, 44, 20, 40, 32, 36, 8, 24, 28, 0, 4, 16, 12},
    {47, 35, 51, 50, 43, 19, 27, 26, 39, 7, 15, 6, 31, 3, 11, 2, 23, 46, 38, 29, 34, 42, 10, 25, 14, 30, 45, 21, 49, 22,
     37, 9, 41, 18, 17, 40, 5, 33, 13, 36, 32, 1, 48, 28, 20, 44, 24, 12, 4, 16, 8, 0},
    {27, 51, 43, 39, 46, 47, 35, 31, 38, 19, 23, 3, 10, 15, 11, 14, 37, 7, 42, 6, 33, 50, 34, 2, 13, 30, 22, 41, 44, 26,
     18, 9, 28, 45, 49, 1, 24, 25, 29, 48, 20, 17, 21, 36, 4, 5, 32, 12, 0, 40, 16, 8},
    {51, 43, 39, 47, 31, 35, 19, 23, 7, 27, 15, 11, 3, 50, 38, 37, 46, 42, 26, 21, 34, 30, 18, 9, 22, 49, 2, 1, 14, 25,
     41, 32, 10, 17, 29, 24, 6, 13, 48, 20, 45, 5, 44, 16, 33, 8, 36, 12, 40, 4, 28, 0},
    {47, 31, 23, 51, 43, 27, 15, 39, 19, 50, 11, 35, 46, 22, 7, 42, 34, 49, 3, 38, 14, 45, 10, 30, 6, 33, 37, 26, 41,
     29, 17, 18, 25, 13, 28, 2, 21, 9, 24, 5, 44, 36, 12, 1, 40, 16, 4, 48, 20, 8, 0, 32},
    {39, 43, 47, 51, 50, 19, 27, 35, 42, 7, 23, 31, 38, 3, 11, 15, 22, 46, 30, 10, 14, 34, 26, 45, 6, 49, 18, 41, 2, 21,
     37, 29, 1, 13, 33, 25, 44, 9, 48, 17, 20, 40, 36, 5, 16, 4, 32, 28, 12, 0, 8, 24},
    {47, 51, 43, 39, 15, 35, 27, 23, 46, 31, 19, 3, 22, 11, 7, 42, 14, 50, 34, 38, 10, 26, 30, 45, 49, 6, 18, 41, 17, 2,
     37, 25, 36, 33, 29, 13, 28, 9, 21, 1, 20, 5, 40, 48, 12, 32, 24, 44, 8, 4, 0, 16},
    {39, 47, 51, 43, 27, 35, 31, 23, 3, 46, 19, 15, 42, 26, 18, 11, 38, 22, 10, 7, 30, 2, 49, 50, 14, 25, 41, 34, 37, 9,
     33, 6, 29, 5, 21, 45, 48, 36, 17, 13, 16, 32, 44, 1, 8, 28, 40, 12, 0, 24, 20, 4},
    {51, 19, 43, 3, 47, 11, 27, 22, 39, 7, 23, 2, 35, 38, 50, 29, 31, 18, 46, 25, 15, 14, 34, 17, 42, 10, 26, 9, 30, 37,
     49, 5, 6, 33, 45, 48, 41, 44, 21, 28, 36, 40, 13, 12, 24, 32, 1, 8, 16, 4, 20, 0},
    {19, 43, 51, 47, 15, 39, 23, 35, 42, 31, 11, 7, 34, 27, 22, 3, 6, 50, 18, 46, 2, 10, 14, 38, 45, 17, 49, 30, 37, 13,
     41, 26, 48, 5, 33, 29, 40, 44, 25, 21, 36, 12, 32, 9, 28, 8, 24, 1, 4, 0, 16, 20},
    {51, 11, 47, 23, 35, 7, 43, 19, 31, 3, 39, 15, 27, 42, 46, 50, 18, 26, 38, 34, 6, 49, 22, 30, 41, 25, 14, 45, 1, 17,
     10, 29, 48, 9, 2, 21, 40, 44, 37, 13, 36, 32, 33, 5, 24, 8, 16, 28, 12, 0, 4, 20},
    {35, 23, 51, 19, 7, 15, 47, 38, 3, 11, 43, 34, 22, 46, 39, 10, 14, 30, 31, 49, 2, 6, 27, 33, 41, 45, 50, 21, 17, 37,
     42, 13, 44, 29, 26, 1, 40, 9, 18, 48, 36, 32, 25, 20, 12, 28, 5, 16, 4, 24, 8, 0},
    {15, 43, 51, 47, 42, 31, 39, 35, 34, 27, 19, 23, 14, 3, 7, 11, 2, 30, 50, 46, 45, 22, 26, 38, 41, 10, 33, 18, 37, 6,
     21, 17, 13, 49, 9, 48, 40, 29, 5, 44, 28, 25, 1, 32, 24, 36, 4, 16, 20, 8, 0, 12},
    {35, 39, 47, 51, 19, 27, 31, 43, 15, 23, 11, 7, 38, 30, 3, 50, 18, 2, 46, 42, 10, 21, 49, 34, 45, 17, 37, 26, 41,
     48, 29, 22, 1, 40, 13, 14, 32, 36, 9, 6, 16, 20, 5, 33, 4, 12, 44, 25, 0, 8, 24, 28},
    {15, 43, 47, 51, 50, 39, 19, 35, 34, 31, 46, 23, 26, 27, 42, 11, 22, 7, 45, 38, 2, 3, 33, 30, 37, 6, 13, 18, 29, 49,
     5, 14, 21, 41, 1, 10, 9, 44, 28, 25, 48, 40, 20, 17, 36, 12, 8, 32, 24, 4, 0, 16},
    {43, 35, 51, 47, 39, 27, 7, 46, 31, 19, 3, 42, 23, 11, 50, 34, 15, 18, 38, 45, 30, 6, 26, 33, 10, 37, 22, 17, 2, 29,
     14, 13, 21, 28, 49, 5, 1, 24, 41, 44, 48, 20, 25, 36, 32, 8, 9, 16, 4, 0, 40, 12},
    {35, 31, 51, 39, 50, 19, 47, 23, 46, 11, 43, 26, 18, 7, 27, 14, 41, 34, 15, 45, 5, 22, 3, 29, 1, 6, 42, 25, 40, 49,
     38, 17, 36, 37, 30, 9, 32, 33, 10, 48, 28, 44, 2, 24, 12, 20, 21, 16, 0, 8, 13, 4},
    {27, 51, 43, 35, 50, 47, 19, 23, 22, 39, 46, 7, 10, 31, 42, 3, 45, 15, 14, 38, 37, 11, 6, 30, 21, 34, 41, 18, 9, 26,
     33, 2, 40, 49, 17, 29, 36, 5, 13, 25, 32, 1, 44, 48, 12, 28, 20, 24, 8, 0, 16, 4},
    {47, 7, 51, 43, 35, 26, 39, 31, 15, 14, 23, 27, 11, 2, 50, 19, 3, 41, 46, 42, 34, 29, 38, 6, 30, 25, 18, 49, 22, 40,
     33, 45, 10, 36, 48, 13, 37, 32, 44, 5, 21, 28, 20, 1, 17, 16, 12, 24, 9, 0, 4, 8},
    {39, 51, 43, 47, 31, 15, 46, 35, 11, 7, 26, 27, 3, 50, 22, 23, 42, 34, 6, 19, 18, 30, 37, 38, 14, 2, 25, 49, 10, 45,
     21, 29, 33, 41, 17, 44, 5, 13, 40, 32, 1, 9, 36, 28, 48, 24, 20, 4, 16, 12, 8, 0},
    {15, 51, 47, 43, 11, 31, 39, 23, 46, 50, 35, 19, 34, 42, 27, 3, 30, 26, 7, 38, 18, 22, 25, 10, 14, 6, 21, 2, 45, 37,
     9, 49, 41, 29, 1, 33, 28, 5, 48, 17, 24, 20, 44, 13, 8, 16, 36, 40, 0, 12, 4, 32},
    {51, 47, 43, 35, 27, 23, 39, 42, 3, 19, 31, 26, 30, 11, 15, 18, 10, 46, 7, 14, 25, 34, 50, 6, 21, 41, 38, 33, 1, 17,
     22, 13, 40, 9, 2, 5, 36, 24, 49, 48, 32, 12, 45, 44, 28, 4, 37, 20, 16, 0, 29, 8},
    {35, 51, 47, 43, 23, 27, 39, 7, 11, 3, 31, 46, 50, 38, 19, 18, 42, 34, 15, 14, 30, 22, 26, 49, 41, 10, 29, 45, 17,
     6, 9, 33, 48, 2, 1, 21, 44, 37, 36, 13, 24, 25, 32, 40, 4, 5, 28, 20, 0, 8, 16, 12},
    {31, 43, 47, 51, 7, 35, 39, 23, 3, 19, 27, 50, 46, 11, 15, 34, 42, 49, 26, 18, 38, 17, 22, 14, 30, 9, 37, 10, 6, 1,
     29, 41, 2, 48, 25, 33, 45, 44, 5, 21, 32, 28, 40, 13, 16, 20, 36, 24, 8, 4, 12, 0},
    {47, 43, 51, 31, 39, 35, 27, 11, 3, 23, 19, 7, 38, 10, 15, 50, 26, 37, 42, 46, 22, 33, 30, 34, 14, 25, 18, 6, 2, 5,
     49, 45, 41, 48, 29, 9, 13, 40, 21, 44, 16, 36, 17, 32, 12, 28, 1, 8, 0, 20, 24, 4},
    {23, 35, 51, 47, 15, 27, 43, 31, 50, 11, 39, 19, 2, 7, 3, 46, 49, 22, 42, 38, 41, 21, 30, 34, 37, 13, 14, 26, 33, 5,
     6, 18, 17, 48, 45, 10, 32, 44, 25, 29, 24, 8, 9, 40, 20, 4, 1, 36, 16, 0, 12, 28},
    {47, 39, 43, 51, 27, 35, 31, 15, 11, 19, 23, 7, 34, 46, 50, 3, 6, 22, 42, 38, 41, 18, 30, 49, 25, 14, 26, 45, 13,
     10, 37, 21, 9, 2, 33, 17, 5, 29, 1, 48, 36, 32, 44, 40, 28, 24, 8, 20, 0, 12, 4, 16},
    {51, 46, 47, 39, 35, 42, 43, 31, 3, 38, 23, 27, 50, 34, 15, 19, 30, 37, 11, 6, 26, 33, 7, 2, 22, 13, 18, 29, 10, 9,
     14, 5, 49, 1, 45, 48, 41, 36, 25, 40, 17, 28, 21, 20, 44, 16, 32, 8, 24, 12, 0, 4},
    {47, 51, 19, 31, 43, 35, 46, 27, 39, 11, 38, 23, 15, 7, 30, 18, 50, 3, 41, 14, 26, 42, 37, 6, 22, 34, 25, 2, 49, 10,
     17, 45, 33, 9, 13, 5, 29, 32, 1, 48, 21, 8, 40, 44, 36, 4, 28, 20, 24, 0, 12, 16},
    {27, 51, 47, 39, 23, 43, 31, 35, 50, 15, 10, 19, 34, 11, 2, 3, 30, 7, 45, 38, 26, 46, 41, 14, 49, 42, 37, 6, 33, 22,
     17, 29, 25, 18, 5, 21, 1, 13, 48, 40, 24, 9, 44, 36, 4, 32, 20, 28, 0, 16, 8, 12},
    {43, 51, 47, 31, 23, 39, 50, 11, 19, 35, 26, 7, 3, 27, 14, 46, 38, 15, 10, 30, 22, 42, 33, 18, 45, 34, 29, 6, 41,
     13, 21, 2, 37, 9, 17, 49, 32, 1, 36, 25, 24, 44, 16, 5, 20, 40, 4, 48, 12, 28, 0, 8},
    {47, 27, 50, 51, 43, 19, 22, 39, 35, 15, 18, 7, 31, 3, 14, 46, 23, 26, 10, 42, 11, 41, 6, 38, 34, 37, 21, 2, 30, 44,
     17, 49, 33, 32, 5, 45, 29, 28, 1, 25, 13, 24, 48, 20, 9, 12, 40, 16, 8, 4, 36, 0}
};

const std::vector<std::vector<int>> example_ddts = {
    {0, 12, 0, 12, 0, 12, 0, 12, 10, 3, 10, 3, 9, 4, 9, 4, 0, 8, 0, 8},
    {11, 2, 11, 2, 10, 2, 10, 2, 9, 4, 9, 4, 11, 1, 11, 1, 11, 2, 11, 2},
    {1, 12, 1, 11, 6, 7, 6, 6, 3, 10, 3, 10, 5, 7, 5, 6, 1, 12, 1, 12},
    {9, 4, 9, 4, 5, 8, 5, 8, 6, 7, 6, 7, 7, 5, 7, 5, 6, 6, 6, 6},
    {7, 5, 7, 5, 6, 7, 6, 7, 2, 10, 2, 11, 5, 7, 5, 7, 3, 9, 3, 9},
    {5, 8, 5, 8, 3, 9, 3, 9, 9, 4, 9, 4, 4, 8, 4, 8, 8, 5, 8, 5},
    {11, 2, 10, 2, 9, 4, 8, 4, 9, 3, 8, 3, 10, 3, 9, 3, 10, 3, 10, 3},
    {9, 4, 8, 4, 6, 7, 6, 7, 8, 5, 8, 5, 5, 8, 5, 8, 7, 5, 7, 5},
    {7, 5, 7, 5, 4, 9, 3, 9, 6, 7, 6, 7, 8, 5, 8, 5, 7, 6, 7, 6},
    {5, 7, 5, 7, 1, 12, 1, 12, 5, 7, 5, 7, 0, 13, 0, 13, 2, 11, 2, 11},
    {8, 5, 8, 5, 6, 7, 6, 7, 8, 5, 8, 5, 6, 7, 6, 7, 6, 6, 6, 6},
    {9, 4, 9, 4, 9, 3, 9, 3, 2, 9, 2, 9, 2, 9, 2, 9, 2, 4, 2, 4},
    {9, 4, 9, 4, 4, 9, 4, 9, 9, 4, 9, 4, 4, 8, 4, 8, 9, 4, 9, 4},
    {9, 2, 10, 2, 7, 6, 7, 6, 6, 7, 6, 7, 8, 2, 8, 2, 8, 2, 9, 2},
    {1, 11, 1, 11, 1, 11, 1, 11, 3, 10, 3, 10, 4, 8, 4, 9, 3, 9, 3, 9},
    {8, 4, 8, 4, 3, 9, 3, 9, 9, 4, 9, 4, 7, 6, 7, 6, 6, 7, 5, 7},
    {9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 3, 9, 3, 9, 3, 9, 3},
    {3, 10, 3, 10, 6, 7, 6, 7, 4, 9, 4, 9, 2, 10, 2, 10, 4, 9, 4, 9},
    {2, 11, 2, 11, 8, 4, 8, 4, 5, 6, 5, 7, 2, 11, 2, 11, 2, 10, 2, 10},
    {3, 10, 3, 10, 8, 5, 7, 5, 5, 8, 5, 8, 4, 8, 5, 8, 4, 9, 4, 9},
    {1, 11, 1, 11, 5, 8, 5, 8, 1, 10, 1, 10, 3, 10, 3, 10, 1, 11, 1, 11},
    {5, 7, 5, 7, 6, 7, 6, 7, 7, 4, 7, 4, 10, 3, 10, 3, 5, 5, 5, 5},
    {7, 5, 7, 5, 4, 9, 4, 9, 6, 7, 6, 7, 7, 6, 7, 6, 6, 7, 6, 7},
    {5, 7, 5, 7, 7, 5, 8, 5, 4, 8, 4, 8, 6, 7, 6, 7, 5, 7, 5, 7},
    {7, 6, 7, 6, 11, 2, 11, 2, 6, 6, 6, 6, 10, 1, 11, 1, 12, 1, 12, 1},
    {5, 8, 5, 8, 6, 7, 6, 7, 8, 5, 8, 5, 8, 5, 8, 5, 7, 5, 7, 5},
    {9, 4, 9, 4, 8, 5, 8, 5, 9, 4, 9, 4, 9, 3, 9, 3, 10, 3, 10, 3},
    {5, 8, 5, 7, 4, 9, 4, 9, 5, 8, 5, 8, 5, 8, 5, 8, 5, 8, 5, 8},
    {6, 7, 6, 7, 8, 4, 8, 4, 5, 7, 5, 7, 7, 6, 7, 6, 7, 6, 7, 6},
    {7, 5, 7, 5, 4, 9, 3, 9, 6, 7, 6, 7, 5, 8, 5, 8, 4, 8, 4, 8},
    {7, 6, 7, 6, 3, 10, 3, 10, 6, 6, 6, 6, 3, 10, 3, 10, 6, 7, 6, 7},
    {6, 7, 6, 7, 6, 7, 6, 7, 3, 10, 3, 10, 6, 7, 6, 7, 4, 9, 4, 9},
    {6, 7, 6, 7, 8, 5, 8, 5, 8, 4, 8, 4, 8, 5, 8, 5, 7, 6, 6, 6},
    {3, 10, 3, 10, 5, 8, 5, 8, 6, 7, 6, 7, 3, 9, 3, 9, 4, 8, 4, 8},
    {9, 4, 9, 4, 6, 6, 6, 6, 4, 8, 4, 8, 9, 4, 9, 4, 8, 5, 8, 5},
    {7, 6, 7, 6, 8, 4, 8, 4, 6, 7, 6, 7, 6, 7, 6, 7, 8, 5, 8, 5},
    {1, 10, 1, 10, 5, 7, 5, 7, 1, 11, 1, 11, 8, 5, 8, 5, 1, 11, 1, 11},
    {6, 6, 6, 6, 8, 5, 8, 5, 10, 3, 10, 3, 7, 6, 7, 6, 9, 3, 9, 3},
    {11, 2, 11, 2, 4, 9, 4, 9, 11, 2, 11, 2, 4, 9, 4, 9, 7, 2, 7, 2},
    {7, 4, 7, 4, 7, 6, 7, 6, 7, 6, 7, 6, 3, 10, 3, 10, 5, 8, 5, 8},
    {9, 3, 9, 3, 8, 5, 8, 5, 9, 4, 9, 4, 10, 3, 9, 3, 8, 5, 8, 4},
    {3, 10, 3, 10, 4, 9, 4, 9, 6, 7, 6, 7, 4, 8, 4, 8, 4, 9, 4, 9},
    {8, 4, 8, 4, 3, 10, 3, 10, 11, 2, 11, 2, 6, 6, 6, 7, 10, 2, 10, 2},
    {2, 11, 2, 11, 8, 5, 8, 5, 2, 11, 2, 11, 5, 8, 5, 8, 2, 8, 2, 8},
    {10, 2, 10, 2, 8, 5, 8, 5, 11, 2, 11, 2, 5, 8, 5, 8, 9, 2, 9, 2},
    {9, 4, 9, 4, 6, 7, 6, 7, 10, 3, 10, 3, 10, 3, 10, 3, 9, 4, 9, 4},
    {4, 8, 4, 8, 4, 9, 4, 9, 9, 4, 9, 4, 6, 7, 6, 7, 5, 8, 5, 8},
    {5, 8, 5, 8, 9, 3, 10, 3, 4, 8, 5, 8, 9, 3, 10, 3, 5, 5, 5, 7},
    {8, 4, 8, 4, 7, 5, 7, 5, 9, 4, 9, 4, 10, 2, 10, 3, 10, 3, 10, 3},
    {6, 7, 6, 7, 9, 4, 9, 4, 9, 4, 9, 4, 6, 6, 6, 6, 7, 5, 7, 5},
    {9, 4, 9, 4, 9, 4, 9, 4, 3, 10, 3, 10, 3, 9, 3, 9, 6, 7, 6, 7},
    {10, 2, 10, 2, 7, 5, 7, 5, 10, 2, 10, 2, 6, 7, 6, 7, 10, 2, 10, 2},
    {9, 4, 9, 4, 9, 4, 9, 4, 7, 6, 7, 6, 7, 6, 7, 5, 8, 5, 8, 5},
    {9, 4, 9, 4, 9, 4, 9, 4, 10, 3, 10, 3, 6, 7, 6, 7, 9, 3, 9, 4},
    {8, 4, 8, 4, 6, 7, 6, 7, 7, 6, 7, 6, 9, 4, 9, 4, 7, 6, 7, 6},
    {8, 5, 8, 5, 5, 7, 5, 7, 9, 4, 9, 4, 5, 7, 5, 7, 8, 4, 8, 4},
    {8, 4, 8, 4, 5, 8, 5, 7, 9, 3, 9, 3, 9, 4, 9, 4, 9, 4, 9, 4},
    {8, 4, 8, 5, 5, 8, 5, 8, 5, 8, 5, 8, 8, 5, 8, 5, 6, 6, 6, 6},
    {11, 3, 10, 2, 9, 5, 8, 4, 10, 3, 10, 3, 9, 4, 9, 4, 10, 3, 10, 3},
    {4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 8, 5, 8, 5, 7, 6, 7, 6},
    {9, 4, 8, 4, 11, 2, 11, 2, 8, 5, 8, 5, 11, 2, 11, 2, 11, 2, 11, 2},
    {7, 6, 7, 6, 8, 5, 8, 5, 9, 4, 9, 4, 5, 8, 5, 8, 8, 5, 8, 5},
    {3, 9, 3, 9, 3, 10, 3, 10, 6, 7, 6, 7, 1, 10, 1, 10, 2, 11, 2, 11},
    {5, 7, 5, 8, 5, 7, 5, 7, 7, 5, 7, 5, 9, 4, 9, 4, 6, 7, 6, 7},
    {6, 6, 6, 6, 8, 5, 8, 5, 5, 7, 5, 7, 6, 7, 6, 7, 6, 7, 6, 7},
    {6, 6, 6, 6, 8, 5, 8, 5, 9, 4, 8, 4, 9, 4, 9, 4, 7, 5, 7, 5},
    {7, 6, 6, 6, 5, 8, 5, 8, 7, 6, 7, 6, 6, 6, 6, 6, 7, 6, 7, 6},
    {10, 3, 10, 2, 4, 9, 4, 9, 10, 2, 10, 1, 5, 7, 5, 7, 4, 6, 4, 2},
    {7, 6, 6, 6, 7, 5, 7, 5, 6, 7, 6, 7, 7, 6, 7, 6, 8, 5, 8, 5},
    {5, 8, 5, 8, 7, 3, 8, 3, 3, 10, 3, 10, 10, 3, 10, 3, 8, 3, 8, 3},
    {8, 5, 8, 5, 6, 7, 6, 7, 7, 6, 6, 6, 5, 8, 5, 8, 6, 7, 6, 7},
    {6, 7, 6, 7, 4, 8, 4, 8, 9, 4, 9, 4, 7, 6, 7, 6, 6, 6, 6, 6},
    {6, 7, 6, 7, 3, 10, 2, 10, 3, 10, 3, 10, 6, 6, 6, 6, 3, 9, 3, 10},
    {10, 3, 10, 3, 3, 10, 3, 10, 10, 3, 10, 3, 4, 9, 4, 9, 3, 8, 3, 8},
    {6, 7, 6, 7, 2, 10, 2, 10, 4, 8, 4, 8, 3, 10, 3, 10, 3, 9, 3, 9},
    {8, 5, 7, 5, 8, 5, 8, 5, 7, 5, 7, 5, 7, 6, 6, 6, 9, 4, 7, 4},
    {6, 7, 6, 7, 9, 3, 9, 3, 11, 0, 11, 0, 12, 0, 12, 0, 8, 0, 8, 0},
    {9, 4, 9, 4, 7, 5, 7, 5, 7, 5, 7, 6, 5, 8, 5, 8, 7, 4, 7, 4},
    {8, 5, 8, 5, 5, 8, 5, 8, 8, 5, 8, 4, 9, 4, 9, 4, 6, 7, 6, 7},
    {7, 4, 7, 4, 5, 8, 5, 8, 9, 4, 9, 4, 10, 3, 10, 3, 7, 5, 7, 5},
    {5, 7, 6, 7, 9, 4, 9, 4, 5, 7, 6, 7, 4, 8, 4, 8, 6, 6, 7, 6},
    {3, 9, 3, 9, 7, 5, 7, 5, 4, 9, 4, 9, 6, 6, 6, 6, 7, 5, 7, 5},
    {6, 6, 5, 6, 9, 4, 9, 4, 7, 6, 7, 6, 2, 11, 2, 10, 4, 7, 4, 7},
    {5, 7, 5, 7, 7, 6, 7, 6, 10, 3, 10, 3, 9, 3, 10, 3, 8, 4, 8, 4},
    {7, 5, 7, 5, 6, 6, 6, 6, 11, 2, 11, 2, 10, 3, 10, 3, 6, 5, 6, 5},
    {9, 3, 9, 3, 9, 3, 9, 3, 8, 5, 8, 5, 4, 9, 4, 9, 5, 8, 5, 8},
    {8, 5, 8, 5, 7, 5, 7, 5, 11, 2, 11, 2, 11, 2, 11, 2, 9, 4, 9, 4},
    {4, 9, 4, 9, 5, 8, 5, 8, 5, 7, 5, 7, 3, 10, 3, 10, 4, 8, 4, 8},
    {9, 4, 9, 4, 7, 6, 7, 6, 6, 7, 6, 7, 8, 5, 8, 5, 6, 6, 6, 6},
    {8, 5, 8, 5, 9, 4, 9, 4, 7, 6, 7, 6, 9, 3, 10, 3, 8, 4, 8, 4},
    {9, 4, 9, 4, 5, 8, 5, 8, 4, 9, 4, 9, 8, 4, 8, 4, 7, 6, 7, 6},
    {3, 10, 3, 10, 2, 11, 2, 11, 6, 7, 6, 7, 4, 8, 4, 8, 4, 9, 4, 9},
    {3, 10, 3, 10, 6, 6, 6, 6, 7, 6, 7, 6, 6, 6, 6, 7, 4, 8, 4, 9},
    {5, 7, 5, 7, 10, 3, 10, 3, 5, 6, 6, 6, 6, 6, 7, 6, 7, 5, 7, 5},
    {4, 8, 4, 8, 7, 5, 8, 5, 5, 8, 5, 8, 5, 8, 5, 8, 7, 6, 7, 5},
    {4, 9, 4, 9, 7, 6, 7, 6, 9, 4, 9, 4, 11, 2, 11, 2, 11, 2, 11, 2},
    {7, 6, 7, 6, 10, 3, 10, 3, 8, 5, 8, 5, 7, 6, 7, 6, 10, 3, 10, 3},
    {9, 4, 9, 4, 10, 3, 10, 3, 7, 5, 7, 5, 6, 7, 6, 7, 9, 4, 9, 4},
    {7, 6, 7, 6, 6, 7, 6, 7, 4, 9, 3, 9, 3, 10, 3, 10, 4, 9, 4, 9},
    {4, 9, 4, 9, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 6, 7, 6, 7}
};
}
#endif //BRIDGE_LIB_BRIDGE_LIB_EXAMPLE_CARDS_DDTS_H_
