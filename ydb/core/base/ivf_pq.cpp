#include "ivf_pq.h"

#include <ydb/library/yql/udfs/common/knn/knn-distance.h>

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/stream/output.h>

#include <limits>

namespace NKikimr::NIvfPq {

namespace NPackedNBitVector {

namespace {

    void ValidateNBits(const size_t nbits) {
        Y_ENSURE(nbits >= 1 && nbits <= 16, "nbits must be between 1 and 16");
    }

    size_t ExtractNBits(const ui8 formatByte) {
        Y_ENSURE((formatByte & FORMAT_BASE) == FORMAT_BASE, "invalid packed nbit vector format byte");
        const size_t nbits = formatByte & ~FORMAT_BASE;
        ValidateNBits(nbits);
        return nbits;
    }

    ui32 Mask(const size_t nbits) {
        return (nbits == 16) ? 0xFFFFu : ((1u << nbits) - 1u);
    }

    ui8 PackElements(const TVector<ui16>& elements, const size_t nbits, IOutputStream* out) {
        const ui32 mask = Mask(nbits);
        ui64 accumulator = 0;
        size_t bitsInAccumulator = 0;

        for (const ui16 val : elements) {
            Y_ENSURE((val & ~mask) == 0, "code value exceeds nbits capacity");
            accumulator |= static_cast<ui64>(val) << bitsInAccumulator;
            bitsInAccumulator += nbits;
            // TODO(raydzast): can be done better, e.g. flush ui16/ui32 when bitsInAccumulator allows
            while (bitsInAccumulator >= 8) {
                out->Write(static_cast<ui8>(accumulator & 0xFF));
                accumulator >>= 8;
                bitsInAccumulator -= 8;
            }
        }

        ui8 pad = 0;
        if (bitsInAccumulator > 0) {
            out->Write(static_cast<ui8>(accumulator & 0xFF));
            pad = static_cast<ui8>(8 - bitsInAccumulator);
        }
        return pad;
    }

    TVector<ui16> UnpackElements(const TStringBuf payload, const size_t nbits, const size_t elementCount) {
        const ui32 mask = Mask(nbits);
        TVector<ui16> result;
        result.reserve(elementCount);

        ui64 accumulator = 0;
        size_t bitsInAccumulator = 0;
        size_t payloadPos = 0;

        for (size_t i = 0; i < elementCount; ++i) {
            while (bitsInAccumulator < nbits && payloadPos < payload.size()) {
                const ui8 currentByte = static_cast<ui8>(payload[payloadPos++]);
                accumulator |= static_cast<ui64>(currentByte) << bitsInAccumulator;
                bitsInAccumulator += 8;
            }
            result.push_back(static_cast<ui16>(accumulator & mask));
            accumulator >>= nbits;
            bitsInAccumulator -= nbits;
        }

        return result;
    }

} // namespace

size_t CalcByteCount(const size_t elementCount, const size_t nbits) {
    return (elementCount * nbits + 7) / 8 + HEADER_SIZE;
}

size_t CalcElementCount(const TStringBuf data) {
    Y_ENSURE(data.size() >= HEADER_SIZE);
    const size_t nBits = ExtractNBits(data.back());
    const ui8 pad = data[data.size() - 2];

    return ((data.size() - HEADER_SIZE) * 8 - pad) / nBits;
}

void Serialize(const TVector<ui16>& elements, const size_t nBits, IOutputStream* out) {
    ValidateNBits(nBits);
    const ui8 pad = PackElements(elements, nBits, out);
    out->Write(pad);
    out->Write(static_cast<ui8>(FORMAT_BASE | nBits));
}

TString Serialize(const TVector<ui16>& elements, const size_t nBits) {
    TStringBuilder builder;
    Serialize(elements, nBits, &builder.Out);
    return builder;
}

TVector<ui16> Deserialize(const TStringBuf data) {
    Y_ENSURE(data.size() >= HEADER_SIZE);
    const size_t nBits = ExtractNBits(data.back());
    const ui8 pad = data[data.size() - 2];

    const TStringBuf payload = data.substr(0, data.size() - HEADER_SIZE);
    const size_t elementCount = ((data.size() - HEADER_SIZE) * 8 - pad) / nBits;
    Y_ENSURE(elementCount * nBits + pad <= payload.size() * 8);

    return UnpackElements(payload, nBits, elementCount);
}

}

// TProductQuantizer {

std::unique_ptr<TProductQuantizer> TProductQuantizer::Create(const ui32 subspaceCount, Ydb::Table::VectorIndexSettings settings, const ui32 maxRounds, TString &error) {
    Y_ENSURE(subspaceCount != 0);

    std::unique_ptr<NKMeans::IClusters> wholeEmbeddingFormatValidator = NKMeans::CreateClusters(settings, 0, error);
    if (!wholeEmbeddingFormatValidator) {
        return nullptr;
    }

    Y_ENSURE(settings.vector_dimension() % subspaceCount == 0);
    settings.set_vector_dimension(settings.vector_dimension() / subspaceCount);
    
    TVector<std::unique_ptr<NKMeans::IClusters>> subquantizers(::Reserve(subspaceCount));
    for (size_t i = 0; i < subspaceCount; ++i) {
        auto clusters = NKMeans::CreateClusters(settings, maxRounds, error);
        if (!clusters) {
            return nullptr;
        }

        subquantizers.push_back(std::move(clusters));
    }

    return std::make_unique<TProductQuantizer>(
        subspaceCount,
        std::move(wholeEmbeddingFormatValidator),
        std::move(subquantizers)
    );
}

TProductQuantizer::TProductQuantizer(
    const ui32 subspaceCount,
    std::unique_ptr<NKMeans::IClusters>&& wholeEmbeddingFormatValidator,
    TVector<std::unique_ptr<NKMeans::IClusters>>&& subquantizers
)
    : SubspaceCount(subspaceCount)
    , WholeEmbeddingFormatValidator_(std::move(wholeEmbeddingFormatValidator))
    , Subquantizers_(std::move(subquantizers))
    , IsSubquantizerFinished_(SubspaceCount, false)
{
    Y_ASSERT(Subquantizers_.size() == SubspaceCount);
}

bool TProductQuantizer::InitializeWithEmbeddings(const TVector<TString> embeddings) {
    TVector<TVector<TString>> subvectorsBySubspace(SubspaceCount, TVector<TString>(embeddings.size()));
    for (size_t rowIdx = 0; rowIdx < embeddings.size(); ++rowIdx) {
        const auto& embedding = embeddings.at(rowIdx);
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

bool TProductQuantizer::SetSubquantizerCentroids(const size_t subspaceIdx, TVector<TString>&& centroids) {
    return Subquantizers_.at(subspaceIdx)->SetClusters(std::move(centroids));
}

bool TProductQuantizer::InitializeWithEmptyRow() {
    for (size_t i = 0; i < SubspaceCount; ++i) {
        if (!Subquantizers_[i]->SetClusters({Subquantizers_[i]->GetEmptyRow()})) {
            return false;
        }
        IsSubquantizerFinished_[i] = true;
    }
    return true;
}

void TProductQuantizer::SetRound(const ui32 round) {
    for (auto& subquantizer : Subquantizers_) {
        subquantizer->SetRound(round);
    }
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

bool TProductQuantizer::Recompute() {
    bool ok = true;
    for (size_t i = 0; i < SubspaceCount; ++i) {
        ok &= Subquantizers_[i]->RecomputeClusters();
    }
    return ok;
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

bool TProductQuantizer::IsSubquantizerFinished(const size_t subspaceIdx) const {
    return IsSubquantizerFinished_.at(subspaceIdx);
}

void TProductQuantizer::SetIsSubquantizerFinished(const size_t subspaceIdx, const bool value) {
    IsSubquantizerFinished_.at(subspaceIdx) = value;
}

void TProductQuantizer::AggregateToSubspaceCluster(const size_t subspaceIdx, const ui32 clusterIdx, const TStringBuf embedding, const ui64 weight) {
    Subquantizers_.at(subspaceIdx)->AggregateToCluster(clusterIdx, embedding, weight);
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

TVector<NTableIndex::NIvfPq::TCell> TProductQuantizer::Quantize(const TStringBuf embedding) const {
    const auto subVectors = NKnnVectorSerialization::SplitEmbedding(embedding, SubspaceCount);
    TVector<NTableIndex::NIvfPq::TCell> code(SubspaceCount);
    for (size_t i = 0; i < SubspaceCount; ++i) {
        const auto& clusters = Subquantizers_[i];
        Y_ENSURE(!clusters->GetClusters().empty(), "Not implemented support for 0 clusters");
        const auto clusterIdx = clusters->FindCluster(subVectors[i]);
        if (!clusterIdx.has_value()) {
            Y_ENSURE(false, "Not found centroid for embedding: " << subVectors[i].Quote());
        }
        code[i] = static_cast<NTableIndex::NIvfPq::TCell>(clusterIdx.value());
    }
    return code;
}

const TVector<TString>& TProductQuantizer::GetSubspaceCentroids(const size_t subspaceIdx) const {
    return Subquantizers_.at(subspaceIdx)->GetClusters();
}

const TVector<ui64>& TProductQuantizer::GetSubspaceClusterSizes(const size_t subspaceIdx) const {
    return Subquantizers_.at(subspaceIdx)->GetClusterSizes();
}

void TProductQuantizer::SetSubspaceClusterSize(const size_t subspaceIdx, const ui32 clusterIdx, const ui64 size) {
    Subquantizers_.at(subspaceIdx)->SetClusterSize(clusterIdx, size);
}

const TVector<ui64>& TProductQuantizer::GetSubspaceNextClusterSizes(const size_t subspaceIdx) const {
    return Subquantizers_.at(subspaceIdx)->GetNextClusterSizes();
}

TString TProductQuantizer::Debug() const {
    TStringBuilder builder;
    for (size_t i = 0; i < SubspaceCount; ++i) {
        builder << " Subspace: " << i << " { " << Subquantizers_[i]->Debug() << " } ";
    }
    return builder;
}

// } TProductQuantizer

namespace {

    constexpr ui64 MinVectorDimension = 1;
    constexpr ui64 MaxVectorDimension = 16384;
    constexpr ui64 MinLevels = 1;
    constexpr ui64 MaxLevels = 16;
    constexpr ui64 MinClusters = 2;
    constexpr ui64 MaxClusters = 2048;
    constexpr ui64 MaxClustersPowLevels = ui64(1) << 30;
    constexpr ui64 MaxVectorDimensionMultiplyClusters = ui64(4) << 20;
    constexpr ui64 MinSubspaces = 1;
    constexpr ui64 MaxSubspaces = 1024;
    constexpr ui64 MinSubspaceBits = 1;
    constexpr ui64 MaxSubspaceBits = 12;
    constexpr ui64 MaxCodebookEntries = ui64(1) << 20;
    
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

    Ydb::Table::VectorIndexSettings_Metric ParseSimilarity(const TString& similarity_, TString& error) {
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

    double ParseDouble(const TString& name, const TString& value, TString& error) {
        double result = 0;
        if (!TryFromString(value, result)) {
            error = TStringBuilder() << "Invalid " << name << ": " << value;
        }
        return result;
    }

} // namespace

bool FillSetting(Ydb::Table::IvfPqSettings& settings, const TString& name, const TString& value, TString& error) {
    error = "";

    const TString nameLower = to_lower(name);
    if (nameLower == "distance") {
        if (settings.mutable_settings()->has_metric()) {
            error = "only one of distance or similarity should be set, not both";
            return false;
        }
        settings.mutable_settings()->set_metric(ParseDistance(value, error));
    } else if (nameLower == "similarity") {
        if (settings.mutable_settings()->has_metric()) {
            error = "only one of distance or similarity should be set, not both";
            return false;
        }
        settings.mutable_settings()->set_metric(ParseSimilarity(value, error));
    } else if (nameLower == "vector_type") {
        settings.mutable_settings()->set_vector_type(ParseVectorType(value, error));
    } else if (nameLower == "vector_dimension") {
        settings.mutable_settings()->set_vector_dimension(ParseUInt32(name, value, MinVectorDimension, MaxVectorDimension, error));
    } else if (nameLower == "kmeans_tree_clusters") {
        settings.mutable_kmeans_tree_settings()->set_clusters(ParseUInt32(name, value, MinClusters, MaxClusters, error));
    } else if (nameLower =="kmeans_tree_levels") {
        settings.mutable_kmeans_tree_settings()->set_levels(ParseUInt32(name, value, MinLevels, MaxLevels, error));
    } else if (nameLower == "kmeans_tree_overlap_clusters") {
        settings.mutable_kmeans_tree_settings()->set_overlap_clusters(ParseUInt32(name, value, MinClusters, MaxClusters, error));
    } else if (nameLower == "kmeans_tree_overlap_ratio") {
        settings.mutable_kmeans_tree_settings()->set_overlap_ratio(ParseDouble(name, value, error));
    } else if (nameLower == "subspaces") {
        settings.set_subspaces(ParseUInt32(name, value, MinSubspaces, MaxSubspaces, error));
    } else if (nameLower == "subspace_bits") {
        settings.set_subspace_bits(ParseUInt32(name, value, MinSubspaceBits, MaxSubspaceBits, error));
    } else{
        error = TStringBuilder() << "Unknown index setting: " << name;
        return false;
    }
    
    return !error;
}

bool ValidateSettings(const Ydb::Table::IvfPqSettings& settings, TString& error) {
    error = "";

    if (auto unknownCount = settings.GetReflection()->GetUnknownFields(settings).field_count(); unknownCount > 0) {
        error = TStringBuilder() << "ivf_pq index settings contain " << unknownCount << " unsupported parameter(s)";
        return false;
    }

    if (!settings.has_settings()) {
        error = "vector index settings should be set";
        return false;
    }
    if (!NKMeans::ValidateSettings(settings.settings(), error)) {
        return false;
    }

    if (settings.settings().vector_type() != Ydb::Table::VectorIndexSettings::VECTOR_TYPE_FLOAT) {
        error = "ivf_pq index supports only vector_type=float";
        return false;
    }

    if (!ValidateSettingInRange("subspaces",
        settings.has_subspaces() ? std::optional<ui64>(settings.subspaces()) : std::nullopt,
        MinSubspaces, MaxSubspaces, error))
    {
        return false;
    }

    if (!ValidateSettingInRange("subspace_bits",
        settings.has_subspace_bits() ? std::optional<ui64>(settings.subspace_bits()) : std::nullopt,
        MinSubspaceBits, MaxSubspaceBits, error))
    {
        return false;
    }

    if (settings.settings().vector_dimension() % settings.subspaces() != 0) {
        error = TStringBuilder() << "vector_dimension (" << settings.settings().vector_dimension()
            << ") must be a multiple of subspaces (" << settings.subspaces() << ")";
        return false;
    }

    const ui64 codebookEntries = ui64(settings.subspaces()) << settings.subspace_bits();
    if (codebookEntries > MaxCodebookEntries) {
        error = TStringBuilder() << "subspaces * 2^subspace_bits (" << settings.subspaces()
            << " * 2^" << settings.subspace_bits() << " = " << codebookEntries
            << ") should be less than or equal to " << MaxCodebookEntries;
        return false;
    }

    if (settings.ivf_type_case() != Ydb::Table::IvfPqSettings::kKmeansTreeSettings) {
        error = "ivf_pq requires kmeans_tree_settings to be set";
        return false;
    }
    const auto& kmeans = settings.kmeans_tree_settings();
    if (!ValidateSettingInRange("kmeans_tree_levels",
        kmeans.has_levels() ? std::optional<ui64>(kmeans.levels()) : std::nullopt,
        MinLevels, MaxLevels, error))
    {
        return false;
    }
    if (!ValidateSettingInRange("kmeans_tree_clusters",
        kmeans.has_clusters() ? std::optional<ui64>(kmeans.clusters()) : std::nullopt,
        MinClusters, MaxClusters, error))
    {
        return false;
    }
    if (kmeans.has_overlap_clusters() && kmeans.overlap_clusters() > kmeans.clusters()) {
        error = "kmeans_tree_overlap_clusters should be less than or equal to kmeans_tree_clusters";
        return false;
    }
    if (kmeans.has_overlap_ratio() && kmeans.overlap_ratio() < 0) {
        error = "kmeans_tree_overlap_ratio should be >= 0";
        return false;
    }

    ui64 clustersPowLevels = 1;
    for (ui64 i = 0; i < kmeans.levels(); ++i) {
        clustersPowLevels *= kmeans.clusters();
        if (clustersPowLevels > MaxClustersPowLevels) {
            error = TStringBuilder() << "Invalid kmeans_tree_clusters^kmeans_tree_levels: "
                << kmeans.clusters() << "^" << kmeans.levels()
                << " should be less than " << MaxClustersPowLevels;
            return false;
        }
    }
    if (ui64(settings.settings().vector_dimension()) * kmeans.clusters() > MaxVectorDimensionMultiplyClusters) {
        error = TStringBuilder() << "Invalid vector_dimension*kmeans_tree_clusters: "
            << settings.settings().vector_dimension() << "*" << kmeans.clusters()
            << " should be less than " << MaxVectorDimensionMultiplyClusters;
        return false;
    }

    return true;
}

namespace {

template <typename T>
TString SubtractCentroidImpl(const TStringBuf embedding, const TStringBuf centroid) {
    NKnnVectorSerialization::TDeserializer<T> embeddingDeserializer(embedding);
    NKnnVectorSerialization::TDeserializer<T> centroidDeserializer(centroid);
    Y_ENSURE(embeddingDeserializer.GetElementCount() == centroidDeserializer.GetElementCount());

    const size_t count = embeddingDeserializer.GetElementCount();
    const T* embeddingData = reinterpret_cast<const T*>(embedding.data());
    const T* centroidData = reinterpret_cast<const T*>(centroid.data());

    TStringBuilder builder;
    NKnnVectorSerialization::TSerializer<T> serializer(&builder.Out);
    for (size_t i = 0; i < count; ++i) {
        serializer.HandleElement(embeddingData[i] - centroidData[i]);
    }
    serializer.Finish();

    return builder;
}

} // namespace

TString SubtractCentroid(const TStringBuf embedding, const TStringBuf centroid) {
    Y_ENSURE(!embedding.empty() && !centroid.empty());
    Y_ENSURE(embedding[embedding.size() - HeaderLen] == centroid[centroid.size() - HeaderLen]);

    switch (static_cast<EFormat>(embedding[embedding.size() - HeaderLen])) {
        case EFormat::FloatVector:
            return SubtractCentroidImpl<float>(embedding, centroid);
        case EFormat::Int8Vector:
            Y_ENSURE(false, "SubtractCentroid is not supported for int8 vectors");
        case EFormat::Uint8Vector:
            Y_ENSURE(false, "SubtractCentroid is not supported for uint8 vectors");
        case EFormat::BitVector:
            Y_ENSURE(false, "SubtractCentroid is not supported for bit vectors");
        default:
            Y_ENSURE(false, "Unknown embedding format");
    }
}

namespace {

float ComputeFloatEmbeddingAdditiveDistance(
    const TStringBuf lhs,
    const TStringBuf rhs,
    Ydb::Table::VectorIndexSettings::Metric metric)
{
    switch (metric) {
        case Ydb::Table::VectorIndexSettings::DISTANCE_EUCLIDEAN: {
            const auto dimension = NKnnVectorSerialization::TDeserializer<float>(lhs).GetElementCount();
            return ::L2SqrDistance(lhs.data(), rhs.data(), dimension);
        }
        case Ydb::Table::VectorIndexSettings::DISTANCE_MANHATTAN: {
            const auto distance = KnnDistance<float>::ManhattanDistance(lhs, rhs);
            Y_ENSURE(distance.has_value(), "Failed to compute manhattan distance");
            return *distance;
        }
        case Ydb::Table::VectorIndexSettings::DISTANCE_COSINE:
        case Ydb::Table::VectorIndexSettings::SIMILARITY_COSINE:
        case Ydb::Table::VectorIndexSettings::SIMILARITY_INNER_PRODUCT:
        case Ydb::Table::VectorIndexSettings::METRIC_UNSPECIFIED:
            Y_ENSURE(false, "Unsupported vector index metric");
    }
}

} // namespace

TString BuildDistanceTable(const TStringBuf residual, const TVector<TVector<TStringBuf>>& codebook,
    const ui32 m, const ui32 nbits, const Ydb::Table::VectorIndexSettings& vectorSettings)
{
    Y_ENSURE(codebook.size() == m);

    const size_t dimension = NKnnVectorSerialization::TDeserializer<float>(residual).GetElementCount();
    Y_ENSURE(dimension % m == 0);
    const size_t subspaceDimension = dimension / m;

    const TVector<TString> subResiduals = NKnnVectorSerialization::SplitEmbedding(residual, m);

    const ui32 k = 1u << nbits;
    const auto metric = vectorSettings.metric();

    TStringBuilder builder;
    NKnnVectorSerialization::TSerializer<float> serializer(&builder.Out);
    for (size_t subspaceIdx = 0; subspaceIdx < m; ++subspaceIdx) {
        const size_t actualCount = codebook[subspaceIdx].size();
        Y_ENSURE(actualCount <= k);

        const TStringBuf subResidual = subResiduals[subspaceIdx];
        for (size_t i = 0; i < actualCount; ++i) {
            const TStringBuf subCentroid = codebook[subspaceIdx][i];

            Y_ENSURE(NKnnVectorSerialization::TDeserializer<float>(subCentroid).GetElementCount() == subspaceDimension);

            const float distance = ComputeFloatEmbeddingAdditiveDistance(subResidual, subCentroid, metric);
            serializer.HandleElement(distance);
        }
        // because k-means algo can produce less that initial k clusters
        for (size_t i = actualCount; i < k; ++i) {
            serializer.HandleElement(std::numeric_limits<float>::infinity());
        }
    }
    serializer.Finish();
    return builder;
}

double ComputeDistanceViaTable(const TStringBuf distanceTable, const TStringBuf codes, const ui32 m, const ui32 nbits) {
    const ui32 k = 1u << nbits;

    Y_ENSURE(NKnnVectorSerialization::TDeserializer<float>(distanceTable).GetElementCount() == m * k);
    Y_ENSURE(NPackedNBitVector::CalcElementCount(codes) == m);

    const float* distances = reinterpret_cast<const float*>(distanceTable.data());
    TVector<ui16> codesData = NPackedNBitVector::Deserialize(codes);

    double sum = 0.0;
    for (ui32 subspaceIdx = 0; subspaceIdx < m; ++subspaceIdx) {
        const ui32 code = codesData[subspaceIdx];
        sum += distances[subspaceIdx * k + code];
    }
    return sum;
}

}