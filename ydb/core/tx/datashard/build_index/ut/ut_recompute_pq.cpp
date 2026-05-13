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

    static VectorIndexSettings MakeVectorSettings(const size_t dimension) {
        VectorIndexSettings result;
        result.set_vector_dimension(dimension);
        result.set_vector_type(VectorIndexSettings::VECTOR_TYPE_UINT8);
        result.set_metric(VectorIndexSettings::DISTANCE_EUCLIDEAN);
        return result;
    }

    template <bool WithParentColumn>
    static void DoBadRequest(Tests::TServer::TPtr server, TActorId sender,
        std::function<void(NKikimrTxDataShard::TEvRecomputePqRequest&)> setupRequest,
        const TString& expectedError, bool expectedErrorSubstring = false)
    {
        auto id = sId.fetch_add(1, std::memory_order_relaxed);
        auto snapshot = CreateVolatileSnapshot(server, {kMainTable});
        auto datashards = GetTableShards(server, sender, kMainTable);
        TTableId tableId = ResolveTableId(server, sender, kMainTable);

        TStringBuilder data;
        TString err;
        UNIT_ASSERT(datashards.size() == 1);

        auto ev = std::make_unique<TEvDataShard::TEvRecomputePqRequest>();
        auto& rec = ev->Record;
        rec.SetId(1);

        rec.SetSeqNoGeneration(id);
        rec.SetSeqNoRound(1);

        rec.SetTabletId(datashards[0]);
        tableId.PathId.ToProto(rec.MutablePathId());

        rec.SetSnapshotTxId(snapshot.TxId);
        rec.SetSnapshotStep(snapshot.Step);

        *rec.MutableSettings() = MakeVectorSettings(4);

        if constexpr (WithParentColumn) {
            rec.SetParent(1);
        }
        rec.SetM(2);
        for (size_t i = 0; i < rec.GetM(); ++i) {
            auto* subquantizers = rec.AddSubquantizers();
            subquantizers->AddCentroids("\x20\x20\2");
            subquantizers->AddCentroids("\x60\x60\2");
        }
        rec.SetEmbeddingColumn("embedding");

        setupRequest(rec);

        NKikimr::DoBadRequest<TEvDataShard::TEvRecomputePqResponse>(server, sender, std::move(ev), datashards[0], expectedError, expectedErrorSubstring);
    }

    static TString DoRecomputePq(Tests::TServer::TPtr server, TActorId sender, std::optional<NTableIndex::NIvfPq::TClusterId> parent,
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

                if (parent) {
                    rec.SetParent(*parent);
                }
                rec.SetM(centroidsBySubquantizer.size());
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

            UNIT_ASSERT_VALUES_EQUAL(reply->Record.SubquantizerResultsSize(), centroidsBySubquantizer.size());

            for (size_t i = 0; i < reply->Record.SubquantizerResultsSize(); ++i) {
                data.Out << "subspace = " << i << "\n";
                const auto& subquantizerResults = reply->Record.GetSubquantizerResults(i);
                UNIT_ASSERT_VALUES_EQUAL(subquantizerResults.CentroidsSize(), subquantizerResults.ClusterSizesSize());
                for (size_t j = 0; j < subquantizerResults.CentroidsSize(); ++j) {
                    data.Out << "\tcluster = " << subquantizerResults.GetCentroids(j)
                             << " size = " << subquantizerResults.GetClusterSizes(j) << "\n";
                }
            }
        }

        return data;
    }

    static void DropTable(Tests::TServer::TPtr server, TActorId sender, const TString& name) {
        ui64 txId = AsyncDropTable(server, sender, "/Root", name);
        WaitTxNotification(server, txId);
    }

    Y_UNIT_TEST_TWIN(BadRequest, WithParentColumn) {
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
        if constexpr (WithParentColumn) {
            CreateBuildTable(server, sender, options, "table-main");
        } else {
            CreateMainTable(server, sender, options);
        }

        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.SetTabletId(0);
        }, TStringBuilder() << "{ <main>: Error: Wrong shard 0 this is " << GetTableShards(server, sender, kMainTable)[0] << " }");
        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            TPathId(0, 0).ToProto(request.MutablePathId());
        }, "{ <main>: Error: Unknown table id: 0 }");

        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.SetSnapshotStep(request.GetSnapshotStep() + 1);
        }, "Error: Unknown snapshot", true);
        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.SetSnapshotTxId(request.GetSnapshotTxId() + 1);
        }, "Error: Unknown snapshot", true);

        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.MutableSettings()->set_vector_type(VectorIndexSettings::VECTOR_TYPE_UNSPECIFIED);
        }, "{ <main>: Error: vector_type should be set }");
        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.MutableSettings()->set_metric(VectorIndexSettings::METRIC_UNSPECIFIED);
        }, "{ <main>: Error: either distance or similarity should be set }");

        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.ClearSubquantizers();
        }, "{ <main>: Error: Invalid subquantizers count: 0 expected 2 }");
        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.ClearSubquantizers();
            for (size_t i = 0; i < request.GetM(); ++i) {
                request.AddSubquantizers();
            }
        }, "{ <main>: Error: Failed to set clusters for subquantizer 0: Clusters have invalid format }");
        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.ClearSubquantizers();
            for (size_t i = 0; i < request.GetM(); ++i) {
                request.AddSubquantizers()->AddCentroids("something");
            }
        }, "{ <main>: Error: Failed to set clusters for subquantizer 0: Clusters have invalid format }");

        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.SetEmbeddingColumn("some");
        }, "{ <main>: Error: Unknown embedding column: some }");

        // test multiple issues:
        DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
            request.ClearSubquantizers();
            request.SetEmbeddingColumn("some");
        }, "[ { <main>: Error: Unknown embedding column: some } { <main>: Error: Invalid subquantizers count: 0 expected 2 } ]");
    }

    Y_UNIT_TEST_TWIN(EmptyTable, WithParentColumn) {
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
        if constexpr (WithParentColumn) {
            CreateBuildTable(server, sender, options, "table-main");
        } else {
            CreateMainTable(server, sender, options);
        }

        const TString recomputed = DoRecomputePq(server, sender,
            WithParentColumn ? std::make_optional(1) : std::nullopt,
            MakeVectorSettings(4),
            {
                {"\x20\x20\2", "\xF0\xF0\2"},
                {"\x60\x60\2", "\x10\x10\2"},
            }
        );
        UNIT_ASSERT_VALUES_EQUAL(recomputed,
            "subspace = 0\n"
            "\tcluster = \x20\x20\2 size = 0\n"
            "\tcluster = \xF0\xF0\2 size = 0\n"
            "subspace = 1\n"
            "\tcluster = \x60\x60\2 size = 0\n"
            "\tcluster = \x10\x10\2 size = 0\n"
        );
    }

    Y_UNIT_TEST(TableWithoutParentColumn) {
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

        ExecSQL(server, sender,
            R"sql(UPSERT INTO `/Root/table-main` (`key`, `embedding`, `data`) VALUES )sql"
            "(1, \"\x10\x10\x70\x70\2\", \"a\"),"
            "(2, \"\x20\x20\x70\x70\2\", \"b\"),"
            "(3, \"\x60\x60\x10\x10\2\", \"c\"),"
            "(4, \"\x70\x70\x10\x10\2\", \"d\");"
        );

        const auto recomputed = DoRecomputePq(server, sender, std::nullopt,
            MakeVectorSettings(4),
            {
                {"\x1F\x1F\2", "\x7A\x7A\2"},
                {"\x6F\x6F\2", "\x20\x20\2"},
            }
        );
        UNIT_ASSERT_VALUES_EQUAL(recomputed,
            "subspace = 0\n"
            "\tcluster = \x18\x18\2 size = 2\n"
            "\tcluster = \x68\x68\2 size = 2\n"
            "subspace = 1\n"
            "\tcluster = \x70\x70\2 size = 2\n"
            "\tcluster = \x10\x10\2 size = 2\n"
        );
    }

    Y_UNIT_TEST(TableWithParentColumn) {
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

        auto create = [&] {
            CreateBuildTable(server, sender, options, "table-main");
        };
        auto recreate = [&] {
            DropTable(server, sender, "table-main");
            create();
        };

        create();
        {
            ExecSQL(server, sender,
                R"sql(UPSERT INTO `/Root/table-main` (`__ydb_parent`, `key`, `embedding`, `data`) VALUES )sql"
                "(1, 1, \"\x10\x10\x70\x70\2\", \"a\"),"
                "(1, 2, \"\x20\x20\x70\x70\2\", \"b\"),"
                "(1, 3, \"\x60\x60\x10\x10\2\", \"c\"),"
                "(1, 4, \"\x70\x70\x10\x10\2\", \"d\");"
            );
            const auto recomputed = DoRecomputePq(server, sender, 1,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(recomputed,
                "subspace = 0\n"
                "\tcluster = \x18\x18\2 size = 2\n"
                "\tcluster = \x68\x68\2 size = 2\n"
                "subspace = 1\n"
                "\tcluster = \x70\x70\2 size = 2\n"
                "\tcluster = \x10\x10\2 size = 2\n"
            );
        }
        recreate();
        {
            ExecSQL(server, sender,
                R"sql(UPSERT INTO `/Root/table-main` (`__ydb_parent`, `key`, `embedding`, `data`) VALUES )sql"
                "(2, 1, \"\x10\x10\x70\x70\2\", \"a\"),"
                "(2, 2, \"\x20\x20\x70\x70\2\", \"b\"),"
                "(2, 3, \"\x60\x60\x70\x70\2\", \"c\"),"
                "(3, 4, \"\x70\x70\x10\x10\2\", \"d\");"
            );
            const auto recomputed = DoRecomputePq(server, sender, 2,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(recomputed,
                "subspace = 0\n"
                "\tcluster = \x18\x18\2 size = 2\n"
                "\tcluster = \x60\x60\2 size = 1\n"
                "subspace = 1\n"
                "\tcluster = \x70\x70\2 size = 3\n"
                "\tcluster = \x20\x20\2 size = 0\n"
            );
        }
        recreate();
        {
            ExecSQL(server, sender,
                R"sql(UPSERT INTO `/Root/table-main` (`__ydb_parent`, `key`, `embedding`, `data`) VALUES )sql"
                "(1, 1, \"\x10\x10\x70\x70\2\", \"a\"),"
                "(2, 2, \"\x20\x20\x70\x70\2\", \"b\"),"
                "(2, 3, \"\x60\x60\x70\x70\2\", \"c\"),"
                "(2, 4, \"\x70\x70\x10\x10\2\", \"d\");"
            );
            const auto recomputed = DoRecomputePq(server, sender, 2,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(recomputed,
                "subspace = 0\n"
                "\tcluster = \x20\x20\2 size = 1\n"
                "\tcluster = \x68\x68\2 size = 2\n"
                "subspace = 1\n"
                "\tcluster = \x70\x70\2 size = 2\n"
                "\tcluster = \x10\x10\2 size = 1\n"
            );
        }
        recreate();
        {
            ExecSQL(server, sender,
                R"sql(UPSERT INTO `/Root/table-main` (`__ydb_parent`, `key`, `embedding`, `data`) VALUES )sql"
                "(0, 1, \"\x10\x10\x70\x70\2\", \"a\"),"
                "(2, 2, \"\x20\x20\x70\x70\2\", \"b\"),"
                "(2, 3, \"\x60\x60\x70\x70\2\", \"c\"),"
                "(3, 4, \"\x70\x70\x10\x10\2\", \"d\");"
            );
            const auto recomputed = DoRecomputePq(server, sender, 2,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(recomputed,
                "subspace = 0\n"
                "\tcluster = \x20\x20\2 size = 1\n"
                "\tcluster = \x60\x60\2 size = 1\n"
                "subspace = 1\n"
                "\tcluster = \x70\x70\2 size = 2\n"
                "\tcluster = \x20\x20\2 size = 0\n"
            );
        }
    }

}

}
