#include "bow_matcher.h"
#include <bitset>

namespace srrg_bench {

BoWMatcher::BoWMatcher(const uint32_t& minimum_distance_between_closure_images_,
                       const bool& add_descriptors_to_database_,
                       const std::string& file_path_vocabulary_,
                       const bool& use_direct_index_,
                       const uint32_t& number_of_direct_index_levels_,
                       const bool& compute_score_only_): _number_of_direct_index_levels(number_of_direct_index_levels_),
                                                         _add_descriptors_to_database(add_descriptors_to_database_),
                                                         _minimum_distance_between_closure_images(minimum_distance_between_closure_images_),
                                                         _compute_score_only(compute_score_only_) {
  _image_numbers.clear();
  _raw_descriptors.clear();
  _bow_descriptors_per_image.clear();
  _bow_features_per_image.clear();

  //ds if no vocabulary is provided
  if (file_path_vocabulary_.empty()) {

    //ds allocate a new vocabulary
    _vocabulary = TemplatedVocabulary<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(10, 6);

    std::cerr << "DBoW2Matcher::DBoW2Matcher|initialized empty vocabulary" << std::endl;
  } else {

    //ds allocate vocabulary - check if vocabulary is presented as text file (check last 3 characters)
    if (file_path_vocabulary_.substr(file_path_vocabulary_.length()-3) == "txt") {
#if DBOW2_DESCRIPTOR_TYPE == 1
      std::cerr << "DBoW2Matcher::DBoW2Matcher|loading vocabulary: " << file_path_vocabulary_ << " from TEXT file" << std::endl;
      _vocabulary.loadFromTextFile(file_path_vocabulary_);
#else
      std::cerr << "DBoW2Matcher::DBoW2Matcher|WARNING: loading vocabulary: " << file_path_vocabulary_ << " from TEXT file is not supported" << std::endl;
#endif
    } else {
      std::cerr << "DBoW2Matcher::DBoW2Matcher|loading vocabulary: " << file_path_vocabulary_ << " from COMPRESSED file" << std::endl;
      _vocabulary.load(file_path_vocabulary_);
    }
    std::cerr << "DBoW2Matcher::DBoW2Matcher|successfully loaded the vocabulary" << std::endl;
  }

  //ds allocate an empty database hooked to the vocabulary
  if (_compute_score_only) {
    _database = TemplatedDatabase<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(_vocabulary, false);
  } else {
    _database = TemplatedDatabase<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(_vocabulary, use_direct_index_, number_of_direct_index_levels_);
  }
}

BoWMatcher::~BoWMatcher() {
  _image_numbers.clear();
  _raw_descriptors.clear();
  _bow_descriptors_per_image.clear();
  _bow_features_per_image.clear();
}

void BoWMatcher::add(const cv::Mat& train_descriptors_,
                     const ImageNumberTrain& image_number_,
                     const std::vector<cv::KeyPoint>& train_keypoints_) {
  TIC(_time_begin);

  //ds bow index to image numbers mapping (since bow always starts numbering at 0)
  _image_numbers.insert(std::make_pair(_image_numbers.size(), image_number_));

  //ds obtain descriptors in dbow2 format
  std::vector<DBOW2_DESCRIPTOR_CLASS::TDescriptor> raw_descriptors(train_descriptors_.rows);
  for (int32_t u = 0; u < train_descriptors_.rows; ++u) {
#if DBOW2_DESCRIPTOR_TYPE == 1
    raw_descriptors[u] = train_descriptors_.row(u);
#else
    _setDescriptor(train_descriptors_.row(u), raw_descriptors[u]);
#endif
  }

  //ds always keep bow formatted descriptors (required for querying descriptor matches)
  _raw_descriptors.push_back(raw_descriptors);

  //ds if there is a vocabulary
  if (!_vocabulary.empty()) {
    _add(raw_descriptors);
  }
  _total_duration_add_seconds += TOC(_time_begin).count();
}

void BoWMatcher::train() {
  TIC(_time_begin);

  //ds if we have to create a vocabulary
  if (_vocabulary.empty()) {

    //ds compute vocabulary
    std::cerr << "BoWMatcher::train|creating a <k: " << _vocabulary.getBranchingFactor() << " L: " << _vocabulary.getDepthLevels()
              << "> vocabulary for " << _raw_descriptors.size() << " descriptor vectors (this might take some time)" << std::endl;
    _vocabulary.create(_raw_descriptors);
    std::cerr << "BoWMatcher::train|created vocabulary of size: " << _vocabulary.size() << std::endl;

    //ds update database with new vocabulary
    _database.setVocabulary(_vocabulary);

    //ds finalize database with added descriptors
    if (_add_descriptors_to_database) {
      std::cerr << "BoWMatcher::train|adding images to database with quantization based on created vocabulary" << std::endl;
      for (uint32_t u = 0; u < _raw_descriptors.size(); ++u) {
        _add(_raw_descriptors[u]);
      }
      std::cerr << "BoWMatcher::train|final database size: " << _database.size() << std::endl;
    } else {
      _raw_descriptors.clear();
    }
  }
  _total_duration_train_seconds += TOC(_time_begin).count();
}

void BoWMatcher::train(const cv::Mat& train_descriptors_,
                       const ImageNumberTrain& image_number_,
                       const std::vector<cv::KeyPoint>& train_keypoints_) {
  //ds TODO purge
}

void BoWMatcher::query(const cv::Mat& query_descriptors_,
                       const ImageNumberQuery& image_number_,
                       const uint32_t& maximum_distance_hamming_,
                       std::vector<ResultImageRetrieval>& closures_) {

  //ds obtain descriptors in dbow2 format
  std::vector<DBOW2_DESCRIPTOR_CLASS::TDescriptor> raw_descriptors(query_descriptors_.rows);
  for (int32_t index = 0; index < query_descriptors_.rows; ++index) {
#if DBOW2_DESCRIPTOR_TYPE == 0 or DBOW2_DESCRIPTOR_TYPE == 2
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
    _database.getVocabulary()->transform(raw_descriptors, bow_descriptors);
  } else {
    _database.getVocabulary()->transform(raw_descriptors, bow_descriptors, bow_features, _number_of_direct_index_levels);
  }
  std::chrono::duration<double> duration_match = TOC(_time_begin);

  //ds association computation duration for a single <query, train> pair (will be included in duration match and add)
  double duration_seconds_association_computation = 0;

  //ds if we added at least one image to the database
  if (_database.size() > 0) {

    //ds result handle
    QueryResults results;

    //ds query database
    TIC(_time_begin);
    _database.query(bow_descriptors, results, _database.size());
    duration_match += TOC(_time_begin);

    //ds association computation timing
    uint64_t number_of_computations_associations = 0;
    std::chrono::duration<double> duration_association_computation(0);

    //ds for each result
    for (const Result& result: results) {

      //ds obtain added reference number
      const ImageNumberTrain& image_number_reference = _image_numbers[result.Id];

      //ds if no descriptor associations have to be computed
      if (_compute_score_only) {

        //ds if we can report the score for precision/recall evaluation
        if ((image_number_ >= _minimum_distance_between_closure_images                      &&
            image_number_reference <= image_number_-_minimum_distance_between_closure_images) ||
            _minimum_distance_between_closure_images == 0                                     ) {

          //ds add the closure
          closures_.push_back(ResultImageRetrieval(result.Score, ImageNumberAssociation(image_number_, image_number_reference)));
        }
      } else {

        //ds readability
        const FeatureVector& feature_vector_query     = bow_features;
        const FeatureVector& feature_vector_reference = _bow_features_per_image[result.Id];

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
                _getMatches_neighratio<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(_raw_descriptors[result.Id],
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

        //ds if we can report the score for precision/recall evaluation
        if ((image_number_ >= _minimum_distance_between_closure_images                      &&
            image_number_reference <= image_number_-_minimum_distance_between_closure_images) ||
            _minimum_distance_between_closure_images == 0                                     ) {

          //ds compute score
          const double score = static_cast<double>(i_old.size())/_raw_descriptors[result.Id].size();

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
  _durations_seconds_query_and_train.push_back(duration_match.count()+duration_seconds_association_computation);
  _total_duration_query_seconds += duration_match.count()+duration_seconds_association_computation;

  //ds sort results in descending score
  std::sort(closures_.begin(), closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_)
      {return a_.number_of_matches_relative > b_.number_of_matches_relative;});
}

void BoWMatcher::clear() {
  BaseMatcher::clear();

  //ds reallocate database
  if (_compute_score_only) {
    _database = TemplatedDatabase<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(_vocabulary, false);
  } else {
    _database = TemplatedDatabase<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>(_vocabulary, true, _number_of_direct_index_levels);
  }
}

void BoWMatcher::_add(const std::vector<DBOW2_DESCRIPTOR_CLASS::TDescriptor>& descriptors_) {

  //ds transform raw descriptors to obtain bow counterparts (required for descriptor wise querying)
  BowVector bow_descriptors;
  FeatureVector bow_features;

  //ds quantize input
  if (_compute_score_only) {
    _database.getVocabulary()->transform(descriptors_, bow_descriptors);
  } else {
    _database.getVocabulary()->transform(descriptors_, bow_descriptors, bow_features, _number_of_direct_index_levels);
  }

  //ds bookkeep for matching
  _bow_descriptors_per_image.push_back(bow_descriptors);
  _bow_features_per_image.push_back(bow_features);

  //ds add current image to database
  _database.add(bow_descriptors, bow_features);
}

#if DBOW2_DESCRIPTOR_TYPE != 1
void BoWMatcher::_setDescriptor(const cv::Mat& descriptor_cv_, DBOW2_DESCRIPTOR_CLASS::TDescriptor& descriptor_dbow2_) const {
  DBOW2_DESCRIPTOR_CLASS::TDescriptor bit_buffer(AUGMENTED_DESCRIPTOR_SIZE_BITS);

  //ds loop over all full bytes
  for (uint32_t u = 0; u < AUGMENTED_DESCRIPTOR_SIZE_BYTES; ++u) {

    //ds get minimal datafrom cv::mat
    const std::bitset<8> bits(descriptor_cv_.at<uchar>(u));

    //ds get bitstring
    for(uint8_t v = 0; v < 8; ++v) {
      bit_buffer[u*8+v] = bits[v];
    }
  }

  //ds check if we have extra bits (less than 1 byte i.e. 8 bits)
  if (AUGMENTED_DESCRIPTOR_SIZE_BITS_EXTRA > 0) {

    //ds get last byte (not fully set)
    const std::bitset<8> bits(descriptor_cv_.at<uchar>(AUGMENTED_DESCRIPTOR_SIZE_BYTES));

    //ds only set the remaining bits
    for(uint32_t v = 0; v < AUGMENTED_DESCRIPTOR_SIZE_BITS_EXTRA; ++v) {
      bit_buffer[AUGMENTED_DESCRIPTOR_SIZE_BITS_IN_BYTES+v] = bits[8-AUGMENTED_DESCRIPTOR_SIZE_BITS_EXTRA+v];
    }
  }

  //ds done
  descriptor_dbow2_ = bit_buffer;
}
#endif

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
