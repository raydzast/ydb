#pragma once

#include "kmeans_clusters.h"

#include <ydb/library/yql/udfs/common/knn/knn-serializer-shared.h>

#include <ydb/public/api/protos/ydb_table.pb.h>

#include <ydb/core/base/table_index.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/string/builder.h>

namespace NKikimr::NIvfPq {

    namespace NPackedNBitVector {

        constexpr size_t HEADER_SIZE = 2;
        constexpr ui8 FORMAT_BASE = 0x80;

        size_t CalcByteCount(const size_t elementCount, const size_t nbits);

        size_t CalcElementCount(const TStringBuf data);

        void Serialize(const TVector<ui16>& elements, const size_t nBits, IOutputStream* out);

        TString Serialize(const TVector<ui16>& elements, const size_t nBits);

        TVector<ui16> Deserialize(const TStringBuf data);
    }

    class TProductQuantizer {
    public:
        const ui32 SubspaceCount;

    private:
        std::unique_ptr<NKMeans::IClusters> WholeEmbeddingFormatValidator_;
        TVector<std::unique_ptr<NKMeans::IClusters>> Subquantizers_;
        TVector<bool> IsSubquantizerFinished_;

    public:
        static std::unique_ptr<TProductQuantizer> Create(const ui32 subspaceCount, Ydb::Table::VectorIndexSettings settings, const ui32 maxRounds, TString& error);

        TProductQuantizer(
            const ui32 subspaceCount,
            std::unique_ptr<NKMeans::IClusters>&& wholeEmbeddingFormatValidator,
            TVector<std::unique_ptr<NKMeans::IClusters>>&& subquantizers
        );

        bool InitializeWithEmbeddings(const TVector<TString> embeddings);
        bool SetSubquantizerCentroids(const size_t subspaceIdx, TVector<TString>&& centroids);

        const TVector<TString>& GetSubspaceCentroids(const size_t subspaceIdx) const;
        const TVector<ui64>& GetSubspaceClusterSizes(const size_t subspaceIdx) const;
        void SetSubspaceClusterSize(const size_t subspaceIdx, const ui32 clusterIdx, const ui64 size);
        const TVector<ui64>& GetSubspaceNextClusterSizes(const size_t subspaceIdx) const;

        TVector<NTableIndex::NIvfPq::TCode> Quantize(const TStringBuf embedding) const;

        void SetRound(const ui32 round);
        bool NextRound();
        void Aggregate(const TStringBuf embedding);
        void AggregateToSubspaceCluster(const size_t subspaceIdx, const ui32 clusterIdx, const TStringBuf embedding, const ui64 weight);
        bool Recompute();

        bool IsSubquantizerFinished(const size_t subspaceIdx) const;
        void SetIsSubquantizerFinished(const size_t subspaceIdx, const bool value);

        bool IsValidEmbedding(const TStringBuf embedding) const;
        void Clear();

        TString Debug() const;
    };


    bool FillSetting(Ydb::Table::IvfPqSettings& settings, const TString& name, const TString& value, TString& error);

    bool ValidateSettings(const Ydb::Table::IvfPqSettings& settings, TString& error);

}
