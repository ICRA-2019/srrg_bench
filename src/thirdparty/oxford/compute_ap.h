// This code originated from: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
// The Oxford Buildings Datastd::set
// James Philbin, Relja Arandjelović and Andrew Zisserman 

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

class OxfordAveragePrecisionUtility {

  OxfordAveragePrecisionUtility() = delete;

  public:

    static std::vector<std::string> load_list(const std::string& fname)
    {
      std::vector<std::string> ret;
      std::ifstream fobj(fname.c_str());
      if (!fobj.good()) { std::cerr << "File " << fname << " not found!\n"; exit(-1); }
      std::string line;
      while (getline(fobj, line)) {
        ret.push_back(line);
      }
      return ret;
    }

    template<class T>
    static std::set<T> vector_to_set(const std::vector<T>& vec)
    { return std::set<T>(vec.begin(), vec.end()); }

    static float compute_ap(const std::set<std::string>& pos, const std::set<std::string>& amb, const std::vector<std::string>& ranked_list)
    {
      float old_recall = 0.0;
      float old_precision = 1.0;
      float ap = 0.0;

      size_t intersect_size = 0;
      size_t i = 0;
      size_t j = 0;
      for ( ; i<ranked_list.size(); ++i) {
        if (amb.count(ranked_list[i])) continue;
        if (pos.count(ranked_list[i])) intersect_size++;

        float recall = intersect_size / (float)pos.size();
        float precision = intersect_size / (j + 1.0);

        ap += (recall - old_recall)*((old_precision + precision)/2.0);

        old_recall = recall;
        old_precision = precision;
        j++;
      }
      return ap;
    }
};
