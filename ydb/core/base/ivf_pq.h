#pragma once

#include "kmeans_clusters.h"

#include <ydb/library/yql/udfs/common/knn/knn-serializer-shared.h>

#include <ydb/public/api/protos/ydb_table.pb.h>

#include <ydb/core/base/table_index.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/string/builder.h>

#include <type_traits>

namespace NKikimr::NIvfPq {

    namespace NPackedNBitVector {

        constexpr size_t HEADER_SIZE = 2;
        constexpr ui8 FORMAT_BASE = 0x80;

        inline size_t ExtractNBits(const ui8 formatByte) {
            return formatByte & ~FORMAT_BASE;
        }

        size_t CalcByteCount(const size_t elementCount, const size_t nbits);

        size_t CalcElementCount(const TStringBuf data);

        template <typename TFrom>
        void Serialize(const TVector<TFrom> elements, const size_t nBits, IOutputStream* out) {
            static_assert(std::is_same<TFrom, ui8>::value, "not implemented");
            Y_ENSURE(nBits == 8 || nBits == 2 /* TODO(raydzast): remove */, "not implemented");

            out->Write(elements.data(), elements.size() * sizeof(TFrom));
            out->Write(static_cast<ui8>(0));
            out->Write(static_cast<ui8>(FORMAT_BASE | nBits));
        }

        template <typename TFrom>
        TString Serialize(const TVector<TFrom> elements, const size_t nBits) {
            TStringBuilder builder;
            Serialize(elements, nBits, &builder.Out);
            return builder;
        }

        template <typename T>
        TVector<T> Deserialize(const TStringBuf data) {
            static_assert(std::is_same<T, ui8>::value, "not implemented");

            const size_t nBits = ExtractNBits(data.back());
            const ui8 pad = data[data.size() - 2];

            Y_ENSURE(nBits == 8, "not implemented");
            Y_ENSURE(pad == 0);

            TVector<T> result(::Reserve(CalcElementCount(data)));
            MemCopy(result.data(), data.data(), data.size());
            return result;
        }
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
