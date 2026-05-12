#include "ut_helpers.h"

#include <ydb/core/testlib/test_client.h>
#include <ydb/core/tx/datashard/ut_common/datashard_ut_common.h>
#include <ydb/public/api/protos/ydb_table.pb.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

namespace NKikimr {
using namespace Tests;
using Ydb::Table::VectorIndexSettings;

static std::atomic<ui64> sId = 1;
static constexpr const char* kMainTable = "/Root/table-main";

Y_UNIT_TEST_SUITE(TTxDataShardRecomputePqScan) {

    static TString DoRecomputePq(Tests::TServer::TPtr server, TActorId sender, NTableIndex::NKMeans::TClusterId parent,
        VectorIndexSettings vectorSettings, TVector<TVector<TString>> centroidsBySubquantizer/*, bool withForeign = false*/)
    {
        auto id = sId.fetch_add(1, std::memory_order_relaxed);
        auto& runtime = *server->GetRuntime();
        auto snapshot = CreateVolatileSnapshot(server, {kMainTable});
        auto datashards = GetTableShards(server, sender, kMainTable);
        TTableId tableId = ResolveTableId(server, sender, kMainTable);

        TStringBuilder data;
        TString err;

        for (auto tid : datashards) {
            auto ev1 = std::make_unique<TEvDataShard::TEvRecomputePqRequest>();
            auto ev2 = std::make_unique<TEvDataShard::TEvRecomputePqRequest>();
            auto fill = [&](std::unique_ptr<TEvDataShard::TEvRecomputePqRequest>& ev) {
                auto& rec = ev->Record;
                rec.SetId(1);

                rec.SetSeqNoGeneration(id);
                rec.SetSeqNoRound(1);

                rec.SetTabletId(tid);
                tableId.PathId.ToProto(rec.MutablePathId());

                rec.SetSnapshotTxId(snapshot.TxId);
                rec.SetSnapshotStep(snapshot.Step);

                *rec.MutableSettings() = vectorSettings;

                // rec.SetSkipOverlapForeign(withForeign);

                rec.SetParent(parent);
                for (const auto& centroids : centroidsBySubquantizer) {
                    NKikimrTxDataShard::TEvRecomputePqRequest::TSubquantizer subquantizer;
                    *subquantizer.MutableCentroids() = {centroids.begin(), centroids.end()};
                    *rec.AddSubquantizers() = std::move(subquantizer);
                }
                rec.SetEmbeddingColumn("embedding");
            };
            fill(ev1);
            fill(ev2);  // TODO(raydzast): who double request?

            runtime.SendToPipe(tid, sender, ev1.release(), 0, GetPipeConfigWithRetries());
            runtime.SendToPipe(tid, sender, ev2.release(), 0, GetPipeConfigWithRetries());

            TAutoPtr<IEventHandle> handle;
            auto reply = runtime.GrabEdgeEventRethrow<TEvDataShard::TEvRecomputePqResponse>(handle);

            NYql::TIssues issues;
            NYql::IssuesFromMessage(reply->Record.GetIssues(), issues);
            UNIT_ASSERT_EQUAL_C(reply->Record.GetStatus(), NKikimrIndexBuilder::EBuildStatus::DONE,
                                issues.ToOneLineString());

            for (size_t i = 0; i < reply->Record.SubquantizerResultsSize(); ++i) {
                data.Out << "subspace = " << i << "\n";
                const auto& subquantizerResults = reply->Record.GetSubquantizerResults(i);
                for (size_t j = 0; j < subquantizerResults.CentroidsSize(); ++j) {
                    data.Out << "\tcluster = " << subquantizerResults.GetCentroids(j)
                             << " size = " << subquantizerResults.GetClusterSizes(j) << "\n";
                }
            }
        }

        return data;
    }

    Y_UNIT_TEST(EmptyTable) {
        TPortManager pm;
        TServerSettings serverSettings(pm.GetPort(2134));
        serverSettings.SetDomainName("Root");

        Tests::TServer::TPtr server = new TServer(serverSettings);
        auto& runtime = *server->GetRuntime();
        auto sender = runtime.AllocateEdgeActor();

        runtime.SetLogPriority(NKikimrServices::TX_DATASHARD, NLog::PRI_DEBUG);
        runtime.SetLogPriority(NKikimrServices::BUILD_INDEX, NLog::PRI_TRACE);

        InitRoot(server, sender);

        TShardedTableOptions options;
        options.Shards(1);
        CreateMainTable(server, sender, options);

        VectorIndexSettings vectorSettings;
        vectorSettings.set_vector_dimension(4);
        vectorSettings.set_vector_type(VectorIndexSettings::VECTOR_TYPE_UINT8);
        vectorSettings.set_metric(VectorIndexSettings::DISTANCE_EUCLIDEAN);

        auto recomputed = DoRecomputePq(server, sender, 0, vectorSettings, {
            {"\x20\x20\2", "\xF0\xF0\2"},
            {"\x60\x60\2", "\x10\x10\2"},
        });
        UNIT_ASSERT_VALUES_EQUAL(recomputed, "cluster = \x22\x36\2 size = 6\ncluster = \x10\x10\2 size = 0\n");
    }
}

}