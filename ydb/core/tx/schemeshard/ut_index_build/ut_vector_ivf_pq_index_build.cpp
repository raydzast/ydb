#include <ydb/core/testlib/basics/runtime.h>
#include <ydb/core/tx/schemeshard/ut_helpers/helpers.h>
#include <ydb/core/tx/schemeshard/ut_helpers/test_env.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NKikimr;
using namespace NSchemeShardUT_Private;

Y_UNIT_TEST_SUITE(VectorIndexIvfPqBuildTest) {
    Y_UNIT_TEST(CreateAndDrop) {
        TTestBasicRuntime runtime;
        TTestEnv env(runtime);
        ui64 txId = 100;

        runtime.SetLogPriority(NKikimrServices::TX_DATASHARD, NLog::PRI_TRACE);
        runtime.SetLogPriority(NKikimrServices::BUILD_INDEX, NLog::PRI_TRACE);

        ui64 tenantSchemeShard = 0;
        TestCreateServerLessDb(runtime, env, txId, tenantSchemeShard);

        TestCreateTable(runtime, tenantSchemeShard, ++txId, "/MyRoot/ServerLessDB", R"(
            Name: "Table"
            Columns { Name: "key"       Type: "Uint32" }
            Columns { Name: "embedding" Type: "String" }
            Columns { Name: "prefix"    Type: "Uint32" }
            Columns { Name: "value"     Type: "String" }
            KeyColumnNames: ["key"]
        )");
        env.TestWaitNotification(runtime, txId, tenantSchemeShard);

        // Write data directly into shards
        WriteVectorTableRows(runtime, tenantSchemeShard, ++txId, "/MyRoot/ServerLessDB/Table", 0, 0, 200);

        TestDescribeResult(DescribePath(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB/Table"),
            {NLs::PathExist, NLs::IndexesCount(0), NLs::PathVersionEqual(3)});

        ui64 buildIndexTx = ++txId;
        TestBuildVectorIvfPqIndex(runtime, buildIndexTx, tenantSchemeShard, "/MyRoot/ServerLessDB", "/MyRoot/ServerLessDB/Table", "index1", {"embedding"});
        env.TestWaitNotification(runtime, buildIndexTx, tenantSchemeShard);

        auto buildIndexOperations = TestListBuildIndex(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB");
        UNIT_ASSERT_VALUES_EQUAL(buildIndexOperations.EntriesSize(), 1);

        auto buildIndexOperation = TestGetBuildIndex(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB", buildIndexTx);
        UNIT_ASSERT_VALUES_EQUAL(buildIndexOperation.GetIndexBuild().GetState(), Ydb::Table::IndexBuildState::STATE_DONE);

        TestDescribeResult(DescribePath(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB/Table"),
            {NLs::PathExist, NLs::IndexesCount(1), NLs::PathVersionEqual(6)});

        TestDescribeResult(DescribePath(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB/Table/index1", true, true, true),
            {NLs::PathExist, NLs::IndexState(NKikimrSchemeOp::EIndexState::EIndexStateReady)});

        TestForgetBuildIndex(runtime, ++txId, tenantSchemeShard, "/MyRoot/ServerLessDB", buildIndexTx);
        buildIndexOperations = TestListBuildIndex(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB");
        UNIT_ASSERT_VALUES_EQUAL(buildIndexOperations.EntriesSize(), 0);

        TestDropTableIndex(runtime, tenantSchemeShard, ++txId, "/MyRoot/ServerLessDB", R"(
            TableName: "Table"
            IndexName: "index1"
        )");
        env.TestWaitNotification(runtime, txId, tenantSchemeShard);

        Cerr << "... rebooting scheme shard" << Endl;
        RebootTablet(runtime, tenantSchemeShard, runtime.AllocateEdgeActor());

        TestDescribeResult(DescribePath(runtime, tenantSchemeShard, "/MyRoot/ServerLessDB/Table"),
            {NLs::PathExist, NLs::IndexesCount(0), NLs::PathVersionEqual(8)});
    }
}
