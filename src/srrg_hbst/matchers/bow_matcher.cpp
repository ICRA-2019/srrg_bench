#include "bow_matcher.h"

namespace srrg_bench {

  BoWMatcher::BoWMatcher(const uint32_t& interspace_image_number_,
                         const uint32_t& minimum_distance_between_closure_images_,
                         const std::string& file_path_vocabulary_,
                         const bool& use_direct_index_,
                         const uint32_t& number_of_direct_index_levels_,
                         const bool& compute_score_only_): _number_of_direct_index_levels(number_of_direct_index_levels_),
                                                           _interspace_image_number(interspace_image_number_),
                                                           _minimum_distance_between_closure_images(minimum_distance_between_closure_images_),
                                                           _compute_score_only(compute_score_only_) {
    _raw_descriptors_per_image.clear();
    _bow_descriptors_per_image.clear();
    _bow_features_per_image.clear();
    _durations_seconds_query_and_train.clear();

    //ds allocate vocabulary and database
#if DBOW2_DESCRIPTOR_TYPE == 0
    std::cerr << "DBoW2Matcher::DBoW2Matcher|loading BRIEF vocabulary: " << file_path_vocabulary_ << std::endl;
    _vocabulary.load(file_path_vocabulary_);
    if (_compute_score_only) {
      _database = new BriefDatabase(_vocabulary, false);
    } else {
      _database = new BriefDatabase(_vocabulary, use_direct_index_, number_of_direct_index_levels_);
    }
#elif DBOW2_DESCRIPTOR_TYPE == 1
    std::cerr << "DBoW2Matcher::DBoW2Matcher|loading ORB vocabulary: " << file_path_vocabulary_ << std::endl;
    _vocabulary.loadFromTextFile(file_path_vocabulary_);
    if (_compute_score_only) {
      _database = new BriefDatabase(_vocabulary, false);
    } else {
      _database = new TemplatedDatabase<FORB::TDescriptor, FORB>(_vocabulary, use_direct_index_, direct_index_levels_);
    }
#endif
  }

  BoWMatcher::~BoWMatcher() {
    _raw_descriptors_per_image.clear();
    _bow_descriptors_per_image.clear();
    _bow_features_per_image.clear();
    delete _database;
  }

  void BoWMatcher::train(const cv::Mat& train_descriptors_,
                         const ImageNumberTrain& image_number_,
                         const std::vector<cv::KeyPoint>& train_keypoints_) {

    //ds bow index to image numbers mapping
    _image_numbers.insert(std::make_pair(_bow_descriptors_per_image.size(), image_number_));

    //ds add current image to database
    TIC(_time_begin);
    _database->add(_bow_descriptors_per_image[image_number_], _bow_features_per_image[image_number_]);
    _durations_seconds_query_and_train.back() += TOC(_time_begin).count();
  }

  void BoWMatcher::query(const cv::Mat& query_descriptors_,
                         const ImageNumberQuery& image_number_,
                         const uint32_t& maximum_distance_hamming_,
                         std::vector<ResultImageRetrieval>& closures_) {

    //ds obtain descriptors in dbow2 format
    std::vector<DBOW2_DESCRIPTOR_CLASS::TDescriptor> raw_descriptors(query_descriptors_.rows);
    for (int32_t index = 0; index < query_descriptors_.rows; ++index) {
#if DBOW2_DESCRIPTOR_TYPE == 0
      _setDescriptor(query_descriptors_.row(index), raw_descriptors[index]);
#elif DBOW2_DESCRIPTOR_TYPE == 1
      raw_descriptors[index] = query_descriptors_.row(index);
#endif
    }

    //ds transform raw descriptors to obtain bow pendants
    BowVector bow_descriptors;
    FeatureVector bow_features;

    //ds transform input
    TIC(_time_begin);
    if (_compute_score_only) {
      _database->getVocabulary()->transform(raw_descriptors, bow_descriptors);
    } else {
      _database->getVocabulary()->transform(raw_descriptors, bow_descriptors, bow_features, _number_of_direct_index_levels);
    }
    std::chrono::duration<double> duration_match = TOC(_time_begin);

    //ds association computation duration for a single <query, train> pair (will be included in duration match and add)
    double duration_seconds_association_computation = 0;

    //ds if we added at least one image to the database
    if (_raw_descriptors_per_image.size() > 0) {

      //ds result handle
      QueryResults results;

      //ds query database
      TIC(_time_begin);
      _database->query(bow_descriptors, results, _raw_descriptors_per_image.size());
      duration_match += TOC(_time_begin);

      //ds association computation timing
      uint64_t number_of_computations_associations = 0;
      std::chrono::duration<double> duration_association_computation(0);

      //ds for each result
      for (const Result& result: results) {
        const ImageNumberTrain& image_number_reference = _image_numbers[result.Id];

        //ds if no descriptor associations have to be computed
        if (_compute_score_only) {

          //ds check if we can report the score for precision/recall evaluation
          if (image_number_ >= _minimum_distance_between_closure_images && image_number_reference <= image_number_-_minimum_distance_between_closure_images) {

            //ds add the closure
            closures_.push_back(ResultImageRetrieval(result.Score, ImageNumberAssociation(image_number_, image_number_reference)));
          }
        } else {

          //ds readability
          const FeatureVector& feature_vector_query     = bow_features;
          const FeatureVector& feature_vector_reference = _bow_features_per_image[image_number_reference];

          //ds obtain descriptor associations
          std::vector<unsigned int> i_old, i_cur;

          FeatureVector::const_iterator old_it        = feature_vector_reference.begin();
          FeatureVector::const_iterator cur_it        = feature_vector_query.begin();
          const FeatureVector::const_iterator old_end = feature_vector_reference.end( );
          const FeatureVector::const_iterator cur_end = feature_vector_query.end();

          //ds start looking for associations: snippet: https://github.com/dorian3d/DLoopDetector/blob/master/include/DLoopDetector/TemplatedLoopDetector.h
          TIC(_time_begin);
          while(old_it != old_end && cur_it != cur_end)
          {
              if(old_it->first == cur_it->first)
              {
                  // compute matches between
                  // features old_it->second of m_image_keys[old_entry] and
                  // features cur_it->second of keys
                  std::vector<unsigned int> i_old_now, i_cur_now;
                  _getMatches_neighratio<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(_raw_descriptors_per_image[image_number_reference],
                                                                                                      old_it->second,
                                                                                                      raw_descriptors,
                                                                                                      cur_it->second,
                                                                                                      i_old_now,
                                                                                                      i_cur_now,
                                                                                                      maximum_distance_hamming_);

                  i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());
                  i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());

                  // move old_it and cur_it forward
                  ++old_it;
                  ++cur_it;
              }
              else if(old_it->first < cur_it->first)
              {
                  // move old_it forward
                  old_it = feature_vector_reference.lower_bound(cur_it->first);
              }
              else
              {
                  // move cur_it forward
                  cur_it = feature_vector_query.lower_bound(old_it->first);
              }
          }
          duration_association_computation += TOC(_time_begin);
          ++number_of_computations_associations;

          //ds check if we can report the score for precision/recall evaluation
          if (image_number_ >= _minimum_distance_between_closure_images && image_number_reference <= image_number_-_minimum_distance_between_closure_images) {

            //ds compute score
            const double score = static_cast<double>(i_old.size())/_raw_descriptors_per_image[image_number_reference].size();

            //ds add the closure
            closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_reference)));
          }
        }
      }

      //ds compute average association duration for one <query, train> pair
      if (number_of_computations_associations > 0) {
        duration_seconds_association_computation = duration_association_computation.count(); ///number_of_computations_associations;
      }
    }

    //ds bookkeep full descriptor information
    _raw_descriptors_per_image.insert(std::make_pair(image_number_, raw_descriptors));
    _bow_descriptors_per_image.insert(std::make_pair(image_number_, bow_descriptors));
    _bow_features_per_image.insert(std::make_pair(image_number_, bow_features));

    _durations_seconds_query_and_train.push_back(duration_match.count()+duration_seconds_association_computation);
  }

  void BoWMatcher::_setDescriptor(const cv::Mat& descriptor_cv_, FBrief::TDescriptor& descriptor_dbow2_) const {
    FBrief::TDescriptor bit_buffer(DESCRIPTOR_SIZE_BITS);

    //ds loop over all bytes
    for (uint32_t u = 0; u < DESCRIPTOR_SIZE_BYTES; ++u) {

      //ds get minimal datafrom cv::mat
      const uchar byte_value = descriptor_cv_.at<uchar>(u);

      //ds get bitstring
      for(uint8_t v = 0; v < 8; ++v) {
        bit_buffer[u*8+v] = (byte_value >> v) & 1;
      }
    }
    descriptor_dbow2_ = bit_buffer;
  }

  template<class TDescriptor, class F>
  void BoWMatcher::_getMatches_neighratio(const std::vector<TDescriptor> &A, const std::vector<unsigned int> &i_A,
                                          const std::vector<TDescriptor> &B, const std::vector<unsigned int> &i_B,
                                          std::vector<unsigned int> &i_match_A, std::vector<unsigned int> &i_match_B,
                                          const uint32_t& maximum_distance_hamming_) const {
    i_match_A.resize(0);
    i_match_B.resize(0);
    i_match_A.reserve( std::min(i_A.size(), i_B.size()) );
    i_match_B.reserve( std::min(i_A.size(), i_B.size()) );

    std::vector<unsigned int>::const_iterator ait, bit;
    unsigned int i, j;
    i = 0;
    for(ait = i_A.begin(); ait != i_A.end(); ++ait, ++i)
    {
      int best_j_now = -1;
      double best_dist_1 = 1e9;
      double best_dist_2 = 1e9;

      j = 0;
      for(bit = i_B.begin(); bit != i_B.end(); ++bit, ++j)
      {
        double d = F::distance(A[*ait], B[*bit]);

        // in i
        if(d < best_dist_1)
        {
          best_j_now = j;
          best_dist_2 = best_dist_1;
          best_dist_1 = d;
        }
        else if(d < best_dist_2)
        {
          best_dist_2 = d;
        }
      }

      if(best_dist_1 / best_dist_2 <= 0.6)
      {
        unsigned int idx_B = i_B[best_j_now];
        bit = find(i_match_B.begin(), i_match_B.end(), idx_B);

        if(bit == i_match_B.end())
        {
          //ds if matching distance is satisfactory
          if(best_dist_1 < maximum_distance_hamming_)
          {
            i_match_B.push_back(idx_B);
            i_match_A.push_back(*ait);
          }
        }
        else
        {
          unsigned int idx_A = i_match_A[ bit - i_match_B.begin() ];
          double d = F::distance(A[idx_A], B[idx_B]);
          if(best_dist_1 < d)
          {
            i_match_A[ bit - i_match_B.begin() ] = *ait;
          }
        }

      }
    }
  }
}
