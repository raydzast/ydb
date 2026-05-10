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

    class TPqClusters {
    public:
        const ui32 M;

    private:
        std::unique_ptr<NKMeans::IClusters> WholeClusters_;
        TVector<std::unique_ptr<NKMeans::IClusters>> ClustersBySubspace_;
        TVector<bool> IsFinishedClusters_;

        TPqClusters(
            const ui32 m,
            std::unique_ptr<NKMeans::IClusters>&& wholeClusters,
            TVector<std::unique_ptr<NKMeans::IClusters>>&& clustersBySubspace
        )
            : M(m)
            , WholeClusters_(std::move(wholeClusters))
            , ClustersBySubspace_(std::move(clustersBySubspace))
            , IsFinishedClusters_(M, false)
        { }

    public:
        static std::optional<TPqClusters> Create(const ui32 m, Ydb::Table::VectorIndexSettings settings, const ui32 maxRounds, TString& error) {
            std::unique_ptr<NKMeans::IClusters> wholeClusters = NKMeans::CreateClusters(settings, 0, error);
            if (!wholeClusters) {
                return std::nullopt;
            }

            Y_ENSURE(settings.vector_dimension() % m == 0);
            settings.set_vector_dimension(settings.vector_dimension() / m);
            TVector<std::unique_ptr<NKMeans::IClusters>> clustersBySubspace;
            for (size_t i = 0; i < m; ++i) {
                auto clusters = NKMeans::CreateClusters(settings, maxRounds, error);
                if (!clusters) {
                    return std::nullopt;
                }

                clustersBySubspace.push_back(std::move(clusters));
            }

            return TPqClusters(m, std::move(wholeClusters), std::move(clustersBySubspace));
        }

        bool NextRound() {
            bool finished = true;
            for (size_t i = 0; i < M; ++i) {
                if (!IsFinishedClusters_[i]) {
                    IsFinishedClusters_[i] = ClustersBySubspace_[i]->NextRound();
                }
                finished &= IsFinishedClusters_[i];
            }
            return finished;
        }

        void ClearClusters() {
            for (auto& c : ClustersBySubspace_) {
                c->Clear();
            }
            IsFinishedClusters_ = TVector<bool>(M, false);
        }
        
        bool SplitAndSetClusters(const TVector<TString> embeddings) {
            TVector<TVector<TString>> subvectorsBySubspace(M, TVector<TString>(embeddings.size()));
            for (size_t rowIdx = 0; rowIdx < embeddings.size(); ++rowIdx) {
                const auto embedding = embeddings.at(rowIdx);
                auto subspaces = NKnnVectorSerialization::SplitEmbedding(embedding, M);
                for (size_t i = 0; i < M; ++i) {
                    subvectorsBySubspace[i][rowIdx] = std::move(subspaces[i]);
                }
            }

            for (size_t i = 0; i < M; ++i) {
                bool ok = ClustersBySubspace_[i]->SetClusters(std::move(subvectorsBySubspace[i]));
                if (!ok) {
                    return false;
                }
                ClustersBySubspace_[i]->SetRound(1);
            }

            return true;
        }

        bool IsExpectedFormat(const TStringBuf embedding) const {
            if (!WholeClusters_->IsExpectedFormat(embedding)) {
                return false;
            }

            const auto subspaces = NKnnVectorSerialization::SplitEmbedding(embedding, M);
            for (size_t i = 0; i < M; ++i) {
                const auto& subEmbedding = subspaces[i];
                if (!ClustersBySubspace_[i]->IsExpectedFormat(subEmbedding)) {
                    return false;
                }
            }

            return true;
        }

        void Aggregate(const TStringBuf embedding) {
            const auto subVectors = NKnnVectorSerialization::SplitEmbedding(embedding, M);
            for (size_t i = 0; i < M; ++i) {
                if (IsFinishedClusters_[i]) {
                    continue;
                }
                const auto& subEmbedding = subVectors[i];
                auto& clusters = *ClustersBySubspace_[i];
                if (auto pos = clusters.FindCluster(subEmbedding); pos) {
                    clusters.AggregateToCluster(*pos, subEmbedding);
                }
            }
        }

        TVector<NTableIndex::NIvfPq::TCode> Encode(const TStringBuf embedding) const {
            const auto subVectors = NKnnVectorSerialization::SplitEmbedding(embedding, M);
            TVector<NTableIndex::NIvfPq::TCode> codes(M);
            for (size_t i = 0; i < M; ++i) {
                const auto& clusters = ClustersBySubspace_[i];
                Y_ENSURE(!clusters->GetClusters().empty(), "Not implemented support for 0 clusters");
                codes[i] = clusters->FindCluster(subVectors[i]).value();
            }
            return codes;
        }

        const TVector<TString>& GetClusters(size_t idx) const {
            return ClustersBySubspace_[idx]->GetClusters();
        }

        TString Debug() const {
            TStringBuilder builder;
            
            for (size_t i = 0; i < M; ++i) {
                builder << " Subspace: " << i << " { Clusters: " << ClustersBySubspace_[i]->Debug() << " } ";
            }

            return builder;
        }
    };


    bool FillSetting(Ydb::Table::IvfPqSettings& settings, const TString& name, const TString& value, TString& error);

    bool ValidateSettings(const Ydb::Table::IvfPqSettings& settings, TString& error);

}
