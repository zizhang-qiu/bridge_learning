//
// Created by qzz on 2023/12/28.
//

#ifndef STAT_MANAGER_H
#define STAT_MANAGER_H
#include <map>
#include <numeric>
#include <string>
#include <vector>
class StatManager {
  public:
  StatManager() = default;

  // Function to add a value associated with a name
  void AddValue(const std::string& name, const double value) {
    // Update min and max values
    UpdateMinMax(name, value);

    // Add the value to the list
    values[name].push_back(value);
  }

  // Function to get the minimum value for a given name
  double GetMin(const std::string& name) const {
    const auto it = min_values.find(name);
    if (it != min_values.end()) {
      return it->second;
    }
    // Return a default value if the name is not found
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Function to get the maximum value for a given name
  double GetMax(const std::string& name) const {
    const auto it = max_values.find(name);
    if (it != max_values.end()) {
      return it->second;
    }
    // Return a default value if the name is not found
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Function to get all values for a given name
  const std::vector<double>& GetAllValues(const std::string& name) const {
    const auto it = values.find(name);
    if (it != values.end()) {
      return it->second;
    }
    // Return an empty vector if the name is not found
    static const std::vector<double> empty_vector;
    return empty_vector;
  }

  // Function to get the average value for a given name
  double GetAverage(const std::string& name) const {
    const auto it = values.find(name);
    if (it != values.end()) {
      const std::vector<double>& value_list = it->second;
      if (!value_list.empty()) {
        const double sum = std::accumulate(value_list.begin(), value_list.end(), 0.0);
        return sum / value_list.size();
      }
    }
    // Return a default value if the name is not found or no values are available
    return std::numeric_limits<double>::quiet_NaN();
  }

  private:
  // Map to store the minimum values for each name
  std::map<std::string, double> min_values;

  // Map to store the maximum values for each name
  std::map<std::string, double> max_values;

  // Map to store all values for each name
  std::map<std::string, std::vector<double>> values;

  // Function to update min and max values for a given name
  void UpdateMinMax(const std::string& name, double value) {
    // Update min value
    const auto min_it = min_values.find(name);
    if (min_it == min_values.end() || value < min_it->second) {
      min_values[name] = value;
    }

    // Update max value
    const auto max_it = max_values.find(name);
    if (max_it == max_values.end() || value > max_it->second) {
      max_values[name] = value;
    }
  }
};
#endif // STAT_MANAGER_H
