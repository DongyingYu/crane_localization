/**
 * @file ORBVocabulary.h
 * @author 
 * @brief
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include "third_party/DBoW2/DBoW2/FORB.h"
#include "third_party/DBoW2/DBoW2/TemplatedVocabulary.h"


typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
            ORBVocabulary;
    
