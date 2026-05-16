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
static constexpr const char* kDatabaseName = "/Root";
static constexpr const char* kMainTable = "/Root/table-main";
static constexpr const char* kPostingTable = "/Root/table-posting";

Y_UNIT_TEST_SUITE(TTxDataShardEncodePqScan) {

    static VectorIndexSettings MakeVectorSettings(const size_t dimension) {
        VectorIndexSettings result;
        result.set_vector_dimension(dimension);
        result.set_vector_type(VectorIndexSettings::VECTOR_TYPE_UINT8);
        result.set_metric(VectorIndexSettings::DISTANCE_EUCLIDEAN);
        return result;
    }

    // template <bool WithParentColumn>
    // static void DoBadRequest(Tests::TServer::TPtr server, TActorId sender,
    //     std::function<void(NKikimrTxDataShard::TEvRecomputePqRequest&)> setupRequest,
    //     const TString& expectedError, bool expectedErrorSubstring = false)
    // {
    //     auto id = sId.fetch_add(1, std::memory_order_relaxed);
    //     auto snapshot = CreateVolatileSnapshot(server, {kMainTable});
    //     auto datashards = GetTableShards(server, sender, kMainTable);
    //     TTableId tableId = ResolveTableId(server, sender, kMainTable);

    //     TStringBuilder data;
    //     TString err;
    //     UNIT_ASSERT(datashards.size() == 1);

    //     auto ev = std::make_unique<TEvDataShard::TEvRecomputePqRequest>();
    //     auto& rec = ev->Record;
    //     rec.SetId(1);

    //     rec.SetSeqNoGeneration(id);
    //     rec.SetSeqNoRound(1);

    //     rec.SetTabletId(datashards[0]);
    //     tableId.PathId.ToProto(rec.MutablePathId());

    //     rec.SetSnapshotTxId(snapshot.TxId);
    //     rec.SetSnapshotStep(snapshot.Step);

    //     *rec.MutableSettings() = MakeVectorSettings(4);

    //     if constexpr (WithParentColumn) {
    //         rec.SetParent(1);
    //     }
    //     rec.SetM(2);
    //     for (size_t i = 0; i < rec.GetM(); ++i) {
    //         auto* subquantizers = rec.AddSubquantizers();
    //         subquantizers->AddCentroids("\x20\x20\2");
    //         subquantizers->AddCentroids("\x60\x60\2");
    //     }
    //     rec.SetEmbeddingColumn("embedding");

    //     setupRequest(rec);

    //     NKikimr::DoBadRequest<TEvDataShard::TEvRecomputePqResponse>(server, sender, std::move(ev), datashards[0], expectedError, expectedErrorSubstring);
    // }

    static TString DoEncodePq(Tests::TServer::TPtr server, TActorId sender, std::optional<NTableIndex::NIvfPq::TClusterId> parent,
        VectorIndexSettings vectorSettings, TVector<TVector<TString>> centroidsBySubquantizer, ui32 maxBatchRows = 50000/*, bool withForeign = false*/)
    {
        auto id = sId.fetch_add(1, std::memory_order_relaxed);
        auto& runtime = *server->GetRuntime();
        auto snapshot = CreateVolatileSnapshot(server, {kMainTable});
        auto datashards = GetTableShards(server, sender, kMainTable);
        TTableId tableId = ResolveTableId(server, sender, kMainTable);

        TString err;
        for (auto tid : datashards) {
            auto ev1 = std::make_unique<TEvDataShard::TEvEncodePqRequest>();
            auto ev2 = std::make_unique<TEvDataShard::TEvEncodePqRequest>();
            auto fill = [&](std::unique_ptr<TEvDataShard::TEvEncodePqRequest>& ev) {
                auto& rec = ev->Record;
                rec.SetId(1);

                rec.SetSeqNoGeneration(id);
                rec.SetSeqNoRound(1);

                rec.SetTabletId(tid);
                tableId.PathId.ToProto(rec.MutablePathId());

                rec.SetSnapshotTxId(snapshot.TxId);
                rec.SetSnapshotStep(snapshot.Step);

                *rec.MutableSettings() = vectorSettings;

                rec.SetM(centroidsBySubquantizer.size());
                rec.SetNBits(8);

                if (parent) {
                    rec.SetParent(*parent);
                }
                for (const auto& centroids : centroidsBySubquantizer) {
                    NKikimrTxDataShard::TEvEncodePqRequest::TSubquantizer subquantizer;
                    *subquantizer.MutableCentroids() = {centroids.begin(), centroids.end()};
                    *rec.AddSubquantizers() = std::move(subquantizer);
                }
                rec.SetEmbeddingColumn("embedding");
                rec.AddDataColumns("data");

                rec.SetDatabaseName(kDatabaseName);
                rec.SetOutputName(kPostingTable);

                rec.MutableScanSettings()->SetMaxBatchRows(maxBatchRows);
            };
            fill(ev1);
            fill(ev2);  // TODO(raydzast): who double request?

            runtime.SendToPipe(tid, sender, ev1.release(), 0, GetPipeConfigWithRetries());
            runtime.SendToPipe(tid, sender, ev2.release(), 0, GetPipeConfigWithRetries());

            TAutoPtr<IEventHandle> handle;
            auto reply = runtime.GrabEdgeEventRethrow<TEvDataShard::TEvEncodePqResponse>(handle);

            NYql::TIssues issues;
            NYql::IssuesFromMessage(reply->Record.GetIssues(), issues);
            UNIT_ASSERT_EQUAL_C(reply->Record.GetStatus(), NKikimrIndexBuilder::EBuildStatus::DONE,
                                issues.ToOneLineString());
        }

        auto posting = ReadShardedTable(server, kPostingTable);
        Cerr << "Posting:" << Endl;
        Cerr << posting.Quote() << Endl;
        return posting;
    }

    static void DropTable(Tests::TServer::TPtr server, TActorId sender, const TString& name) {
        ui64 txId = AsyncDropTable(server, sender, "/Root", name);
        WaitTxNotification(server, txId);
    }

    // Y_UNIT_TEST_TWIN(BadRequest, WithParentColumn) {
    //     TPortManager pm;
    //     TServerSettings serverSettings(pm.GetPort(2134));
    //     serverSettings.SetDomainName("Root");

    //     Tests::TServer::TPtr server = new TServer(serverSettings);
    //     auto& runtime = *server->GetRuntime();
    //     auto sender = runtime.AllocateEdgeActor();

    //     runtime.SetLogPriority(NKikimrServices::TX_DATASHARD, NLog::PRI_DEBUG);
    //     runtime.SetLogPriority(NKikimrServices::BUILD_INDEX, NLog::PRI_TRACE);

    //     InitRoot(server, sender);

    //     TShardedTableOptions options;
    //     options.Shards(1);
    //     if constexpr (WithParentColumn) {
    //         CreateBuildTable(server, sender, options, "table-main");
    //     } else {
    //         CreateMainTable(server, sender, options);
    //     }

    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.SetTabletId(0);
    //     }, TStringBuilder() << "{ <main>: Error: Wrong shard 0 this is " << GetTableShards(server, sender, kMainTable)[0] << " }");
    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         TPathId(0, 0).ToProto(request.MutablePathId());
    //     }, "{ <main>: Error: Unknown table id: 0 }");

    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.SetSnapshotStep(request.GetSnapshotStep() + 1);
    //     }, "Error: Unknown snapshot", true);
    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.SetSnapshotTxId(request.GetSnapshotTxId() + 1);
    //     }, "Error: Unknown snapshot", true);

    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.MutableSettings()->set_vector_type(VectorIndexSettings::VECTOR_TYPE_UNSPECIFIED);
    //     }, "{ <main>: Error: vector_type should be set }");
    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.MutableSettings()->set_metric(VectorIndexSettings::METRIC_UNSPECIFIED);
    //     }, "{ <main>: Error: either distance or similarity should be set }");

    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.ClearSubquantizers();
    //     }, "{ <main>: Error: Invalid subquantizers count: 0 expected 2 }");
    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.ClearSubquantizers();
    //         for (size_t i = 0; i < request.GetM(); ++i) {
    //             request.AddSubquantizers();
    //         }
    //     }, "{ <main>: Error: Failed to set clusters for subquantizer 0: Clusters have invalid format }");
    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.ClearSubquantizers();
    //         for (size_t i = 0; i < request.GetM(); ++i) {
    //             request.AddSubquantizers()->AddCentroids("something");
    //         }
    //     }, "{ <main>: Error: Failed to set clusters for subquantizer 0: Clusters have invalid format }");

    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.SetEmbeddingColumn("some");
    //     }, "{ <main>: Error: Unknown embedding column: some }");

    //     // test multiple issues:
    //     DoBadRequest<WithParentColumn>(server, sender, [](NKikimrTxDataShard::TEvRecomputePqRequest& request) {
    //         request.ClearSubquantizers();
    //         request.SetEmbeddingColumn("some");
    //     }, "[ { <main>: Error: Unknown embedding column: some } { <main>: Error: Invalid subquantizers count: 0 expected 2 } ]");
    // }

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
        CreatePqPostingTable(server, sender, options);

        const TString posting = DoEncodePq(server, sender,
            WithParentColumn ? std::make_optional(1) : std::nullopt,
            MakeVectorSettings(4),
            {
                {"\x20\x20\2", "\xF0\xF0\2"},
                {"\x60\x60\2", "\x10\x10\2"},
            }
        );

        UNIT_ASSERT_VALUES_EQUAL(posting, "");
    }

    // Y_UNIT_TEST(TableWithoutParentColumn) {
    //     TPortManager pm;
    //     TServerSettings serverSettings(pm.GetPort(2134));
    //     serverSettings.SetDomainName("Root");

    //     Tests::TServer::TPtr server = new TServer(serverSettings);
    //     auto& runtime = *server->GetRuntime();
    //     auto sender = runtime.AllocateEdgeActor();

    //     runtime.SetLogPriority(NKikimrServices::TX_DATASHARD, NLog::PRI_DEBUG);
    //     runtime.SetLogPriority(NKikimrServices::BUILD_INDEX, NLog::PRI_TRACE);

    //     InitRoot(server, sender);

    //     TShardedTableOptions options;
    //     options.Shards(1);
    //     CreateMainTable(server, sender, options);
    //     CreatePqPostingTable(server, sender, options);

    //     ExecSQL(server, sender,
    //         R"sql(UPSERT INTO `/Root/table-main` (`key`, `embedding`, `data`) VALUES )sql"
    //         "(1, \"\x10\x10\x33\x08\2\", \"a\"),"
    //         "(2, \"\x2E\x2E\x76\x7F\2\", \"b\"),"
    //         "(3, \"\x60\x6E\x80\x60\2\", \"c\"),"
    //         "(4, \"\x70\x70\x00\x00\2\", \"d\");"
    //     );

    //     const auto posting = DoEncodePq(server, sender, std::nullopt,
    //         MakeVectorSettings(4),
    //         {
    //             {"\x1F\x1F\2", "\x7A\x7A\2"},
    //             {"\x6F\x6F\2", "\x20\x20\2"},
    //         }
    //     );
    //     UNIT_ASSERT_VALUES_EQUAL(posting,
    //         "key = 1, __ydb_codes = \0\1\0\x88, data = a\n"
    //         "key = 2, __ydb_codes = \0\0\0\x88, data = b\n"
    //         "key = 3, __ydb_codes = \1\0\0\x88, data = c\n"
    //         "key = 4, __ydb_codes = \1\1\0\x88, data = d\n"_sb
    //     );
    // }

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
            CreatePqPostingTable(server, sender, options);
        };
        auto recreate = [&] {
            DropTable(server, sender, "table-main");
            DropTable(server, sender, "table-posting");
            create();
        };

        create();
        {
            ExecSQL(server, sender,
                R"sql(UPSERT INTO `/Root/table-main` (`__ydb_parent`, `key`, `embedding`, `data`) VALUES )sql"
                "(1, 1, \"\x10\x10\x70\x70\2\", \"a\"),"
                "(1, 2, \"\x20\x20\x10\x10\2\", \"b\"),"
                "(1, 3, \"\x60\x60\x10\x10\2\", \"c\"),"
                "(1, 4, \"\x70\x70\x70\x70\2\", \"d\");"
            );
            const auto posting = DoEncodePq(server, sender, 1,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(posting,
                "__ydb_parent = 1, key = 1, __ydb_codes = \0\0\0\x88, data = a\n"
                "__ydb_parent = 1, key = 2, __ydb_codes = \0\1\0\x88, data = b\n"
                "__ydb_parent = 1, key = 3, __ydb_codes = \1\1\0\x88, data = c\n"
                "__ydb_parent = 1, key = 4, __ydb_codes = \1\0\0\x88, data = d\n"_sb
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
            const auto posting = DoEncodePq(server, sender, 2,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(posting,
                "__ydb_parent = 2, key = 1, __ydb_codes = \0\0\0\x88, data = a\n"
                "__ydb_parent = 2, key = 2, __ydb_codes = \0\0\0\x88, data = b\n"
                "__ydb_parent = 2, key = 3, __ydb_codes = \1\0\0\x88, data = c\n"_sb
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
            const auto posting = DoEncodePq(server, sender, 2,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(posting,
                "__ydb_parent = 2, key = 2, __ydb_codes = \0\0\0\x88, data = b\n"
                "__ydb_parent = 2, key = 3, __ydb_codes = \1\0\0\x88, data = c\n"
                "__ydb_parent = 2, key = 4, __ydb_codes = \1\1\0\x88, data = d\n"_sb
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
            const auto posting = DoEncodePq(server, sender, 2,
                MakeVectorSettings(4),
                {
                    {"\x1F\x1F\2", "\x7A\x7A\2"},
                    {"\x6F\x6F\2", "\x20\x20\2"},
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(posting,
                "__ydb_parent = 2, key = 2, __ydb_codes = \0\0\0\x88, data = b\n"
                "__ydb_parent = 2, key = 3, __ydb_codes = \1\0\0\x88, data = c\n"_sb
            );
        }
    }

}

}
