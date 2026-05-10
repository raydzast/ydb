#include "ivf_pq.h"

#include <util/string/builder.h>
#include <util/string/cast.h>

namespace NKikimr::NIvfPq {

namespace {
    constexpr ui64 MinVectorDimension = 1;
    constexpr ui64 MaxVectorDimension = 16384;
    constexpr ui64 MinLevels = 1;
    constexpr ui64 MaxLevels = 16;
    constexpr ui64 MinClusters = 2;
    constexpr ui64 MaxClusters = 2048;
    [[maybe_unused]] constexpr ui64 MaxClustersPowLevels = ui64(1) << 30;
    [[maybe_unused]] constexpr ui64 MaxVectorDimensionMultiplyClusters = ui64(4) << 20; // 4 bytes per dimension for float vector type ~= 16 MB
    constexpr ui64 MinPqM = 1;
    constexpr ui64 MaxPqM = 128;
    constexpr ui64 MinPqNBits = 1;
    constexpr ui64 MaxPqNBits = 128;
    
    bool ValidateSettingInRange(const TString& name, std::optional<ui64> value, ui64 minValue, ui64 maxValue, TString& error) {
        if (!value.has_value()) {
            error = TStringBuilder() << name << " should be set";
            return false;
        }

        if (minValue <= *value && *value <= maxValue) {
            return true;
        }

        error = TStringBuilder() << "Invalid " << name << ": " << *value << " should be between " << minValue << " and " << maxValue;
        return false;
    };

    Ydb::Table::VectorIndexSettings_Metric ParseDistance(const TString& distance_, TString& error) {
        const TString distance = to_lower(distance_);
        if (distance == "cosine")
            return Ydb::Table::VectorIndexSettings::DISTANCE_COSINE;
        else if (distance == "manhattan")
            return Ydb::Table::VectorIndexSettings::DISTANCE_MANHATTAN;
        else if (distance == "euclidean")
            return Ydb::Table::VectorIndexSettings::DISTANCE_EUCLIDEAN;
        else {
            error = TStringBuilder() << "Invalid distance: " << distance_;
            return Ydb::Table::VectorIndexSettings::METRIC_UNSPECIFIED;
        }
    };

    [[maybe_unused]] Ydb::Table::VectorIndexSettings_Metric ParseSimilarity(const TString& similarity_, TString& error) {
        const TString similarity = to_lower(similarity_);
        if (similarity == "cosine")
            return Ydb::Table::VectorIndexSettings::SIMILARITY_COSINE;
        else if (similarity == "inner_product")
            return Ydb::Table::VectorIndexSettings::SIMILARITY_INNER_PRODUCT;
        else {
            error = TStringBuilder() << "Invalid similarity: " << similarity_;
            return Ydb::Table::VectorIndexSettings::METRIC_UNSPECIFIED;
        }
    };

    Ydb::Table::VectorIndexSettings_VectorType ParseVectorType(const TString& vectorType_, TString& error) {
        const TString vectorType = to_lower(vectorType_);
        if (vectorType == "float")
            return Ydb::Table::VectorIndexSettings::VECTOR_TYPE_FLOAT;
        else if (vectorType == "uint8")
            return Ydb::Table::VectorIndexSettings::VECTOR_TYPE_UINT8;
        else if (vectorType == "int8")
            return Ydb::Table::VectorIndexSettings::VECTOR_TYPE_INT8;
        else if (vectorType == "bit")
            return Ydb::Table::VectorIndexSettings::VECTOR_TYPE_BIT;
        else {
            error = TStringBuilder() << "Invalid vector_type: " << vectorType_;
            return Ydb::Table::VectorIndexSettings::VECTOR_TYPE_UNSPECIFIED;
        }
    }

    ui32 ParseUInt32(const TString& name, const TString& value, ui64 minValue, ui64 maxValue, TString& error) {
        ui32 result = 0;
        if (!TryFromString(value, result)) {
            error = TStringBuilder() << "Invalid " << name << ": " << value;
            return result;
        }
        ValidateSettingInRange(name, result, minValue, maxValue, error);
        return result;
    }

    [[maybe_unused]] double ParseDouble(const TString& name, const TString& value, TString& error) {
        double result = 0;
        if (!TryFromString(value, result)) {
            error = TStringBuilder() << "Invalid " << name << ": " << value;
        }
        return result;
    }

}

std::optional<TProductQuantizer> TProductQuantizer::Create(const ui32 subspaceCount, Ydb::Table::VectorIndexSettings settings, const ui32 maxRounds, TString &error) {
    std::unique_ptr<NKMeans::IClusters> wholeEmbeddingFormatValidator = NKMeans::CreateClusters(settings, 0, error);
    if (!wholeEmbeddingFormatValidator) {
        return std::nullopt;
    }

    Y_ENSURE(settings.vector_dimension() % subspaceCount == 0);
    settings.set_vector_dimension(settings.vector_dimension() / subspaceCount);
    TVector<std::unique_ptr<NKMeans::IClusters>> subquantizers;
    for (size_t i = 0; i < subspaceCount; ++i) {
        auto clusters = NKMeans::CreateClusters(settings, maxRounds, error);
        if (!clusters) {
            return std::nullopt;
        }

        subquantizers.push_back(std::move(clusters));
    }

    return TProductQuantizer(
        subspaceCount,
        std::move(wholeEmbeddingFormatValidator),
        std::move(subquantizers)
    );
}

bool TProductQuantizer::InitializeWithEmbeddings(const TVector<TString> embeddings) {
    TVector<TVector<TString>> subvectorsBySubspace(SubspaceCount, TVector<TString>(embeddings.size()));
    for (size_t rowIdx = 0; rowIdx < embeddings.size(); ++rowIdx) {
        const auto embedding = embeddings.at(rowIdx);
        auto subspaces = NKnnVectorSerialization::SplitEmbedding(embedding, SubspaceCount);
        for (size_t i = 0; i < SubspaceCount; ++i) {
            subvectorsBySubspace[i][rowIdx] = std::move(subspaces[i]);
        }
    }

    for (size_t i = 0; i < SubspaceCount; ++i) {
        bool ok = Subquantizers_[i]->SetClusters(std::move(subvectorsBySubspace[i]));
        if (!ok) {
            return false;
        }
        Subquantizers_[i]->SetRound(1);
    }

    return true;
}

bool TProductQuantizer::NextRound() {
    bool finished = true;
    for (size_t i = 0; i < SubspaceCount; ++i) {
        if (!IsSubquantizerFinished_[i]) {
            IsSubquantizerFinished_[i] = Subquantizers_[i]->NextRound();
        }
        finished &= IsSubquantizerFinished_[i];
    }
    return finished;
}

void TProductQuantizer::Clear() {
    for (auto& c : Subquantizers_) {
        c->Clear();
    }
    IsSubquantizerFinished_ = TVector<bool>(SubspaceCount, false);
}

bool TProductQuantizer::IsValidEmbedding(const TStringBuf embedding) const {
    if (!WholeEmbeddingFormatValidator_->IsExpectedFormat(embedding)) {
        return false;
    }

    const TVector<TString> subembeddings = NKnnVectorSerialization::SplitEmbedding(embedding, SubspaceCount);
    for (size_t i = 0; i < SubspaceCount; ++i) {
        const auto& subembedding = subembeddings[i];
        if (!Subquantizers_[i]->IsExpectedFormat(subembedding)) {
            return false;
        }
    }

    return true;
}

void TProductQuantizer::Aggregate(const TStringBuf embedding) {
    const auto subVectors = NKnnVectorSerialization::SplitEmbedding(embedding, SubspaceCount);
    for (size_t i = 0; i < SubspaceCount; ++i) {
        if (IsSubquantizerFinished_[i]) {
            continue;
        }
        const auto& subEmbedding = subVectors[i];
        auto& clusters = *Subquantizers_[i];
        if (auto pos = clusters.FindCluster(subEmbedding); pos) {
            clusters.AggregateToCluster(*pos, subEmbedding);
        }
    }
}

TVector<NTableIndex::NIvfPq::TCode> TProductQuantizer::Quantize(const TStringBuf embedding) const {
    const auto subVectors = NKnnVectorSerialization::SplitEmbedding(embedding, SubspaceCount);
    TVector<NTableIndex::NIvfPq::TCode> codes(SubspaceCount);
    for (size_t i = 0; i < SubspaceCount; ++i) {
        const auto& clusters = Subquantizers_[i];
        Y_ENSURE(!clusters->GetClusters().empty(), "Not implemented support for 0 clusters");
        codes[i] = clusters->FindCluster(subVectors[i]).value();
    }
    return codes;
}
const TVector<TString>& TProductQuantizer::GetSubspaceCentroids(const size_t subspaceIdx) const {
    return Subquantizers_.at(subspaceIdx)->GetClusters();
}

TString TProductQuantizer::Debug() const {
    TStringBuilder builder;
    for (size_t i = 0; i < SubspaceCount; ++i) {
        builder << " Subspace: " << i << " { " << Subquantizers_[i]->Debug() << " } ";
    }
    return builder;
}


namespace NPackedNBitVector {

    size_t CalcByteCount(const size_t elementCount, const size_t nbits) {
        return (elementCount * nbits + 7) / 8 + HEADER_SIZE;
    }

    size_t CalcElementCount(const TStringBuf data) {
        const size_t nBits = ExtractNBits(data.back());
        const ui8 pad = data[data.size() - 2];

        return ((data.size() - HEADER_SIZE) * 8 - pad) / nBits; 
    }

}

bool FillSetting([[maybe_unused]] Ydb::Table::IvfPqSettings& settings, [[maybe_unused]] const TString& name, [[maybe_unused]] const TString& value, TString& error) {
    error = "";

    const TString nameLower = to_lower(name);
    if (nameLower == "distance") {
        if (settings.mutable_settings()->has_metric()) {
            error = "only one of distance or similarity should be set, not both";
            return false;
        }
        settings.mutable_settings()->set_metric(ParseDistance(value, error));
    // TODO(raydzast): is similarity compatible with pq?
    // } else if (nameLower == "similarity") {
    //     if (settings.mutable_settings()->has_metric()) {
    //         error = "only one of distance or similarity should be set, not both";
    //         return false;
    //     }
    //     settings.mutable_settings()->set_metric(ParseSimilarity(value, error));
    } else if (nameLower =="vector_type") {
        settings.mutable_settings()->set_vector_type(ParseVectorType(value, error));
    } else if (nameLower =="vector_dimension") {
        settings.mutable_settings()->set_vector_dimension(ParseUInt32(name, value, MinVectorDimension, MaxVectorDimension, error));
    } else if (nameLower == "kmeans_tree_clusters") {
        settings.mutable_kmeans_tree_settings()->set_clusters(ParseUInt32(name, value, MinClusters, MaxClusters, error));
    } else if (nameLower =="kmeans_tree_levels") {
        settings.mutable_kmeans_tree_settings()->set_levels(ParseUInt32(name, value, MinLevels, MaxLevels, error));
    } else if (nameLower == "kmeans_tree_overlap_clusters") {
        settings.mutable_kmeans_tree_settings()->set_overlap_clusters(ParseUInt32(name, value, MinClusters, MaxClusters, error));
    } else if (nameLower == "kmeans_tree_overlap_ratio") {
        settings.mutable_kmeans_tree_settings()->set_overlap_ratio(ParseDouble(name, value, error));
    } else if (nameLower == "pq_m") {
        settings.set_pq_m(ParseUInt32(name, value, MinPqM, MaxPqM, error));
    } else if (nameLower == "pq_nbits") {
        settings.set_pq_nbits(ParseUInt32(name, value, MinPqNBits, MaxPqNBits, error));
    } else{
        error = TStringBuilder() << "Unknown index setting: " << name;
        return false;
    }

    //TODO(raydzast): implement
    
    return !error;
}


bool ValidateSettings([[maybe_unused]] const Ydb::Table::IvfPqSettings& settings, [[maybe_unused]] TString& error) {
    // TODO(raydzast): implement
    return true;
}

}