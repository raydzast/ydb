#include <library/cpp/testing/unittest/registar.h>
#include <ydb/core/kqp/ut/common/kqp_ut_common.h>

#include <ydb/public/sdk/cpp/include/ydb-cpp-sdk/client/table/table.h>

namespace NKikimr {
    namespace NKqp {

        using namespace NYdb;
        using namespace NYdb::NTable;

        Y_UNIT_TEST_SUITE(KqpVectorIndexesIvfPq) {

            TKikimrRunner Kikimr() {
                TTestLogSettings logSettings = TTestLogSettings()
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::FLAT_TX_SCHEMESHARD, NActors::NLog::EPriority::PRI_DEBUG)
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::TX_DATASHARD, NActors::NLog::EPriority::PRI_DEBUG)
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::TABLET_EXECUTOR, NActors::NLog::EPriority::PRI_DEBUG)
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::BUILD_INDEX, NActors::NLog::EPriority::PRI_DEBUG);

                NKikimrConfig::TFeatureFlags featureFlags;
                featureFlags.SetEnableIvfPqIndex(true);
                featureFlags.SetEnableTruncateTable(true);

                auto settings = TKikimrSettings()
                                    .SetFeatureFlags(featureFlags)
                                    .SetVerbose(true)
                                    .SetLogSettings(std::move(logSettings));
                settings.AppConfig.MutableTableServiceConfig()->SetBackportMode(NKikimrConfig::TTableServiceConfig_EBackportMode_All);
                return TKikimrRunner(settings);
            }

            void CreateMain(NQuery::TQueryClient& db) {
                const TString query = R"sql(
                    CREATE TABLE `/Root/main` (
                        `Key` Uint64 NOT NULL,
                        `Embedding` String,
                        `Data` String,
                        PRIMARY KEY (Key)
                    );
                )sql";
                const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
            }

            TResultSet ReadIndex(NQuery::TQueryClient& db, const char* table = "indexImplTable", const char* columns = "*") {
                TString query = Sprintf(R"sql(
                    SELECT %s FROM `/Root/main/ivf_pq_index/%s`;
                )sql", columns, table);
                auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                return result.GetResultSet(0);
            }

            TResultSet ReadMain(NQuery::TQueryClient& db, const char* columns = "*") {
                TString query = Sprintf(R"sql(
                    SELECT %s FROM `/Root/main`;
                )sql", columns);
                auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                return result.GetResultSet(0);
            }

            void UpsertFixture(NQuery::TQueryClient& db) {
                const TString query = R"(
                    UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                    VALUES
                        (1u, Untag(Knn::ToBinaryStringFloat([1.0f, 1.0f, 2.0f, 2.0f]), "FloatVector"), "one"),
                        (2u, Untag(Knn::ToBinaryStringFloat([3.0f, 3.0f, 4.0f, 4.0f]), "FloatVector"), "two"),
                        (3u, Untag(Knn::ToBinaryStringFloat([5.0f, 5.0f, 6.0f, 6.0f]), "FloatVector"), "three"),
                        (4u, Untag(Knn::ToBinaryStringFloat([7.0f, 7.0f, 8.0f, 8.0f]), "FloatVector"), "four"),
                        (5u, Untag(Knn::ToBinaryStringFloat([1.5f, 0.5f, 5.5f, 6.5f]), "FloatVector"), "five"),
                        (6u, Untag(Knn::ToBinaryStringFloat([2.5f, 2.5f, 4.5f, 3.5f]), "FloatVector"), "six"),
                        (7u, Untag(Knn::ToBinaryStringFloat([4.5f, 5.5f, 7.5f, 8.5f]), "FloatVector"), "seven"),
                        (8u, Untag(Knn::ToBinaryStringFloat([6.5f, 7.5f, 2.5f, 1.5f]), "FloatVector"), "eight"),
                        (9u, Untag(Knn::ToBinaryStringFloat([2.0f, 2.0f, 1.0f, 1.0f]), "FloatVector"), "nine"),
                        (10u, Untag(Knn::ToBinaryStringFloat([4.0f, 4.0f, 3.0f, 3.0f]), "FloatVector"), "ten"),
                        (11u, Untag(Knn::ToBinaryStringFloat([7.2f, 7.2f, 7.8f, 7.8f]), "FloatVector"), "eleven");
                )";
                const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
            }

            void AddIndex(NQuery::TQueryClient& db, bool covered) {
                const TString query = Sprintf(R"sql(
                    ALTER TABLE `/Root/main`
                        ADD INDEX `ivf_pq_index`
                            GLOBAL SYNC
                            USING vector_ivf_pq
                            ON (`Embedding`)%s
                            WITH (
                                vector_dimension=4,
                                vector_type=float,
                                distance=euclidean,
                                kmeans_tree_clusters=2,
                                kmeans_tree_levels=1,
                                subspaces=2,
                                subspace_bits=2
                            );
                )sql", (covered ? " COVER (`Data`)" : ""));
                const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
            }

            Y_UNIT_TEST(AddIndexIvfPq) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                UpsertFixture(db);
                AddIndex(db, false);

                const auto level = ReadIndex(db, "indexImplLevelTable", "`__ydb_parent`, `__ydb_id`");
                UNIT_ASSERT_VALUES_EQUAL(level.RowsCount(), 2u);
                THashSet<ui64> levelIds;
                {
                    TResultSetParser parser(level);
                    while (parser.TryNextRow()) {
                        UNIT_ASSERT_VALUES_EQUAL(parser.ColumnParser(0).GetUint64(), 0u);
                        levelIds.insert(parser.ColumnParser(1).GetUint64());
                    }
                }
                UNIT_ASSERT(levelIds.contains(9223372036854775809u));
                UNIT_ASSERT(levelIds.contains(9223372036854775810u));

                const auto codebook = ReadIndex(db, "indexImplCodebookTable", "`__ydb_parent`");
                UNIT_ASSERT_VALUES_EQUAL(codebook.RowsCount(), 8u);

                const auto posting = ReadIndex(db, "indexImplPostingTable", "`__ydb_parent`, `Key`");
                UNIT_ASSERT_VALUES_EQUAL(posting.RowsCount(), 11u);
            }

            Y_UNIT_TEST(AnnQueryE2e) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                UpsertFixture(db);
                AddIndex(db, false);

                THashSet<ui64> bruteForceKeys;
                {
                    const TString query = R"sql(
                        $target = Untag(Knn::ToBinaryStringFloat([0.0f, 0.0f, 0.0f, 0.0f]), "FloatVector");
                        SELECT `Key` FROM `/Root/main`
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    TResultSetParser parser(result.GetResultSet(0));
                    while (parser.TryNextRow()) {
                        const auto key = parser.ColumnParser(0).GetUint64();
                        bruteForceKeys.insert(key);
                    }
                }

                THashSet<ui64> annKeys;
                {
                    const TString query = R"(
                        PRAGMA ydb.KMeansTreeSearchTopSize = "2";
                        $target = Untag(Knn::ToBinaryStringFloat([0.0f, 0.0f, 0.0f, 0.0f]), "FloatVector");
                        SELECT `Key` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    TResultSetParser parser(result.GetResultSet(0));
                    while (parser.TryNextRow()) {
                        const auto key = parser.ColumnParser(0).GetUint64();
                        annKeys.insert(key);
                    }
                }

                ui32 overlap = 0;
                for (const auto key : annKeys) {
                    if (bruteForceKeys.contains(key)) {
                        ++overlap;
                    }
                }
                UNIT_ASSERT_C(overlap >= 1, "ANN recall@3 should overlap brute-force top-3 on the fixture");
            }

            Y_UNIT_TEST(EmptyTableBuild) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                AddIndex(db, false);

                CompareYsonUnordered(
                    R"([[0u;9223372036854775809u]])",
                    FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "`__ydb_parent`, `__ydb_id`")));

                CompareYsonUnordered(
                    R"([[0u;0u;0u];[0u;1u;0u]])",
                    FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable",
                        "`__ydb_parent`, `__ydb_subspace`, `__ydb_cell`")));

                CompareYson(
                    R"([])",
                    FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "`__ydb_parent`")));

                {
                    const TString query = R"sql(
                        $target = Untag(Knn::ToBinaryStringFloat([0.0f, 0.0f, 0.0f, 0.0f]), "FloatVector");
                        SELECT `Key` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    CompareYson(R"([])", FormatResultSetYson(result.GetResultSet(0)));
                }
            }

            Y_UNIT_TEST(UnderpopulatedBuild) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);

                {
                    const TString query = R"(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        VALUES
                            (1u, Untag(Knn::ToBinaryStringFloat([1.0f, 1.0f, 1.0f, 1.0f]), "FloatVector"), "one"),
                            (2u, Untag(Knn::ToBinaryStringFloat([2.0f, 2.0f, 2.0f, 2.0f]), "FloatVector"), "two"),
                            (3u, Untag(Knn::ToBinaryStringFloat([3.0f, 3.0f, 3.0f, 3.0f]), "FloatVector"), "three");
                    )";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                AddIndex(db, false);

                CompareYsonUnordered(
                    R"([[0u;0u;0u];[0u;0u;1u];[0u;0u;2u];[0u;1u;0u];[0u;1u;1u];[0u;1u;2u]])",
                    FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable",
                        "`__ydb_parent`, `__ydb_subspace`, `__ydb_cell`")));

                CompareYsonUnordered(
                    R"([[1u];[2u];[3u]])",
                    FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "`Key`")));

                {
                    const TString query = R"(
                        PRAGMA ydb.KMeansTreeSearchTopSize = "2";
                        $target = Untag(Knn::ToBinaryStringFloat([0.0f, 0.0f, 0.0f, 0.0f]), "FloatVector");
                        SELECT `Key` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 10;
                    )";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    UNIT_ASSERT_VALUES_EQUAL(result.GetResultSet(0).RowsCount(), 3u);
                }
            }

            // With EnableIvfPqIndex on, logical rewrite must produce ProductQuantizationBuildDistanceTable in the AST.
            Y_UNIT_TEST(AnnQueryRewritePlan) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetTableClient();
                auto session = db.CreateSession().GetValueSync().GetSession();

                {
                    const TString query = R"sql(
                        CREATE TABLE `/Root/main` (
                            `Key` Uint64,
                            `Embedding` String,
                            `Data` String,
                            PRIMARY KEY (Key)
                        );
                    )sql";
                    const auto result = session.ExecuteSchemeQuery(query).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                {
                    const TString query = R"sql(
                        ALTER TABLE `/Root/main`
                            ADD INDEX `ivf_pq_index`
                                GLOBAL SYNC
                                USING vector_ivf_pq
                                ON (`Embedding`)
                                WITH (
                                    vector_dimension=4,
                                    vector_type=float,
                                    distance=euclidean,
                                    kmeans_tree_clusters=2,
                                    kmeans_tree_levels=1,
                                    subspaces=2,
                                    subspace_bits=2
                                );
                    )sql";
                    const auto result = session.ExecuteSchemeQuery(query).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                {
                    // 4 float zeros + FloatVector format tag (0x01) at the end.
                    const TString query = R"(
                        $target = "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01";
                        SELECT `Key` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )";
                    const auto result = session.ExplainDataQuery(query).ExtractValueSync();
                    UNIT_ASSERT_C(result.IsSuccess(), result.GetIssues().ToString());
                    const auto& ast = result.GetAst();
                    UNIT_ASSERT_C(ast.find("ProductQuantizationBuildDistanceTable") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("indexImplCodebookTable") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("indexImplPostingTable") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("IvfPqDistanceTables") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("Collect") != std::string::npos, ast);
                }

                {
                    const TString query = R"(
                        $target = "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01";
                        SELECT `Key`, `Data` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )";
                    const auto result = session.ExplainDataQuery(query).ExtractValueSync();
                    UNIT_ASSERT_C(result.IsSuccess(), result.GetIssues().ToString());
                    const auto& ast = result.GetAst();
                    UNIT_ASSERT_C(ast.find("indexImplPostingTable") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("\"/Root/main\"") != std::string::npos, ast);
                }
            }


            void DoTestDelete(bool covered, const TString& deleteQuery,
                const TString& expectedMainKeysYson, const TString& expectedPostingKeysYson,
                const TMaybe<TString>& returningExpected = Nothing())
            {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                UpsertFixture(db);
                AddIndex(db, covered);

                const TString levelBefore = FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*"));
                const TString codebookBefore = FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*"));

                {
                    const auto result = db.ExecuteQuery(deleteQuery, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    if (returningExpected) {
                        CompareYson(*returningExpected, FormatResultSetYson(result.GetResultSet(0)));
                    }
                }

                CompareYsonUnordered(expectedMainKeysYson, FormatResultSetYson(ReadMain(db, "`Key`")));
                CompareYsonUnordered(expectedPostingKeysYson, FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "`Key`")));
                CompareYsonUnordered(levelBefore, FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*")));
                CompareYsonUnordered(codebookBefore, FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*")));
            }

            Y_UNIT_TEST_TWIN(DeletePk, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` WHERE `Key`=5u;)",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])");
            }

            Y_UNIT_TEST_TWIN(DeleteFilter, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` WHERE `Data`="five";)",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])");
            }

            Y_UNIT_TEST_TWIN(DeleteOn, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` ON SELECT 5u AS `Key`;)",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])");
            }

            Y_UNIT_TEST_TWIN(DeletePkReturning, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` WHERE `Key`=5u RETURNING `Data`, `Key`;)",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    TMaybe<TString>(R"([[["five"];5u]])"));
            }

            Y_UNIT_TEST_TWIN(DeleteFilterReturning, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` WHERE `Data`="five" RETURNING `Data`, `Key`;)",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    TMaybe<TString>(R"([[["five"];5u]])"));
            }

            Y_UNIT_TEST_TWIN(DeleteOnReturning, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` ON SELECT 5u AS `Key` RETURNING `Data`, `Key`;)",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    TMaybe<TString>(R"([[["five"];5u]])"));
            }

            Y_UNIT_TEST_TWIN(DeleteAll, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main`;)",
                    R"([])",
                    R"([])");
            }

            Y_UNIT_TEST_TWIN(DeleteNoMatch, Covered) {
                DoTestDelete(Covered,
                    R"(DELETE FROM `/Root/main` WHERE `Key`=999u;)",
                    R"([[1u];[2u];[3u];[4u];[5u];[6u];[7u];[8u];[9u];[10u];[11u]])",
                    R"([[1u];[2u];[3u];[4u];[5u];[6u];[7u];[8u];[9u];[10u];[11u]])");
            }

            void DoTestUpdate(bool covered, const TString& updateQuery,
                const TString& expectedMainYson, const TString& expectedPostingKeysYson,
                const TMaybe<TString>& returningExpected = Nothing())
            {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                UpsertFixture(db);
                AddIndex(db, covered);

                const TString levelBefore = FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*"));
                const TString codebookBefore = FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*"));

                {
                    const auto result = db.ExecuteQuery(updateQuery, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    if (returningExpected) {
                        CompareYson(*returningExpected, FormatResultSetYson(result.GetResultSet(0)));
                    }
                }

                CompareYsonUnordered(expectedMainYson, FormatResultSetYson(ReadMain(db, "`Key`, `Data`")));
                CompareYsonUnordered(expectedPostingKeysYson, FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "`Key`")));
                CompareYsonUnordered(levelBefore, FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*")));
                CompareYsonUnordered(codebookBefore, FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*")));
            }

            static constexpr const char* EmbOne =
                R"(Untag(Knn::ToBinaryStringFloat([1.0f, 1.0f, 1.0f, 1.0f]), "FloatVector"))";

            static constexpr const char* MainOriginal =
                R"([[1u;["one"]];[2u;["two"]];[3u;["three"]];[4u;["four"]];[5u;["five"]];[6u;["six"]];[7u;["seven"]];[8u;["eight"]];[9u;["nine"]];[10u;["ten"]];[11u;["eleven"]]])";
            static constexpr const char* MainRow5New =
                R"([[1u;["one"]];[2u;["two"]];[3u;["three"]];[4u;["four"]];[5u;["new"]];[6u;["six"]];[7u;["seven"]];[8u;["eight"]];[9u;["nine"]];[10u;["ten"]];[11u;["eleven"]]])";
            static constexpr const char* MainRow5NullData =
                R"([[1u;["one"]];[2u;["two"]];[3u;["three"]];[4u;["four"]];[5u;#];[6u;["six"]];[7u;["seven"]];[8u;["eight"]];[9u;["nine"]];[10u;["ten"]];[11u;["eleven"]]])";
            static constexpr const char* MainAllX =
                R"([[1u;["X"]];[2u;["X"]];[3u;["X"]];[4u;["X"]];[5u;["X"]];[6u;["X"]];[7u;["X"]];[8u;["X"]];[9u;["X"]];[10u;["X"]];[11u;["X"]]])";
            static constexpr const char* PostingAllKeys =
                R"([[1u];[2u];[3u];[4u];[5u];[6u];[7u];[8u];[9u];[10u];[11u]])";
            static constexpr const char* PostingWo5 = 
                R"([[1u];[2u];[3u];[4u];[6u];[7u];[8u];[9u];[10u];[11u]])";

            static constexpr const char* MainWith12 =
                R"([[1u;["one"]];[2u;["two"]];[3u;["three"]];[4u;["four"]];[5u;["five"]];[6u;["six"]];[7u;["seven"]];[8u;["eight"]];[9u;["nine"]];[10u;["ten"]];[11u;["eleven"]];[12u;["twelve"]]])";
            static constexpr const char* MainWith12Emb =
                R"([[1u;["one"]];[2u;["two"]];[3u;["three"]];[4u;["four"]];[5u;["five"]];[6u;["six"]];[7u;["seven"]];[8u;["eight"]];[9u;["nine"]];[10u;["ten"]];[11u;["eleven"]];[12u;#]])";
            static constexpr const char* PostingWith12 =
                R"([[1u];[2u];[3u];[4u];[5u];[6u];[7u];[8u];[9u];[10u];[11u];[12u]])";

            Y_UNIT_TEST_TWIN(Update, Covered) {
                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        SET `Data` = "new"
                        WHERE `Key` = 5u;
                    )sql",
                    MainRow5New, PostingAllKeys);

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        SET `Data` = "new"
                        WHERE `Data` = "five";
                    )sql",
                    MainRow5New, PostingAllKeys);

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        ON SELECT 5u AS `Key`, "new" AS `Data`;
                    )sql",
                    MainRow5New, PostingAllKeys);

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        SET `Data` = "new"
                        WHERE `Key` = 5u
                        RETURNING `Data`, `Key`;
                    )sql",
                    MainRow5New, PostingAllKeys,
                    TMaybe<TString>(R"([[["new"];5u]])"));

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        SET `Data` = "new"
                        WHERE `Data` = "five"
                        RETURNING `Data`, `Key`;
                    )sql",
                    MainRow5New, PostingAllKeys,
                    TMaybe<TString>(R"([[["new"];5u]])"));

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        ON SELECT 5u AS `Key`, "new" AS `Data`
                        RETURNING `Data`, `Key`;
                    )sql",
                    MainRow5New, PostingAllKeys,
                    TMaybe<TString>(R"([[["new"];5u]])"));

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        SET `Data` = "X";
                    )sql",
                    MainAllX, PostingAllKeys);

                DoTestUpdate(Covered,
                    R"sql(
                        UPDATE `/Root/main`
                        SET `Data` = "new"
                        WHERE `Key` = 999u;
                    )sql",
                    MainOriginal, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        SET `Embedding` = %s
                        WHERE `Key` = 5u;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        SET `Embedding` = %s
                        WHERE `Data` = "five";
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        ON SELECT 5u AS `Key`, %s AS `Embedding`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        SET `Embedding` = %s
                        WHERE `Key` = 5u
                        RETURNING `Data`, `Key`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys,
                    TMaybe<TString>(R"([[["five"];5u]])"));

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        SET `Embedding` = %s
                        WHERE `Data` = "five"
                        RETURNING `Data`, `Key`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys,
                    TMaybe<TString>(R"([[["five"];5u]])"));

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        ON SELECT 5u AS `Key`, %s AS `Embedding`
                        RETURNING `Data`, `Key`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys,
                    TMaybe<TString>(R"([[["five"];5u]])"));

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        SET `Embedding` = %s;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPDATE `/Root/main`
                        SET `Embedding` = %s
                        WHERE `Key` = 999u;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);
            }

            Y_UNIT_TEST_TWIN(Upsert, Covered) {
                DoTestUpdate(Covered,
                    R"sql(
                        UPSERT INTO `/Root/main` (`Key`, `Data`)
                        VALUES (5u, "new");
                    )sql",
                    MainRow5New, PostingAllKeys);

                DoTestUpdate(Covered,
                    R"sql(
                        UPSERT INTO `/Root/main` (`Key`, `Data`)
                        VALUES (5u, "new")
                        RETURNING `Data`, `Key`;
                    )sql",
                    MainRow5New, PostingAllKeys,
                    TMaybe<TString>(R"([[["new"];5u]])"));

                DoTestUpdate(Covered,
                    R"sql(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        SELECT `Key`, `Embedding`, "X" AS `Data` FROM `/Root/main`;
                    )sql",
                    MainAllX, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`)
                        VALUES (5u, %s);
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`)
                        VALUES (5u, %s)
                        RETURNING `Data`, `Key`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys,
                    TMaybe<TString>(R"([[["five"];5u]])"));

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        SELECT `Key`, %s AS `Embedding`, `Data` FROM `/Root/main`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);
            }

            Y_UNIT_TEST_TWIN(Replace, Covered) {
                DoTestUpdate(Covered,
                    R"sql(
                        REPLACE INTO `/Root/main` (`Key`, `Data`)
                        VALUES (5u, "new");
                    )sql",
                    MainRow5New, PostingWo5);

                DoTestUpdate(Covered,
                    R"sql(
                        REPLACE INTO `/Root/main` (`Key`, `Data`)
                        VALUES (5u, "new")
                        RETURNING `Data`, `Key`;
                    )sql",
                    MainRow5New, PostingWo5,
                    TMaybe<TString>(R"([[["new"];5u]])"));

                DoTestUpdate(Covered,
                    R"sql(
                        REPLACE INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        SELECT `Key`, `Embedding`, "X" AS `Data` FROM `/Root/main`;
                    )sql",
                    MainAllX, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        REPLACE INTO `/Root/main` (`Key`, `Embedding`)
                        VALUES (5u, %s);
                    )sql", EmbOne),
                    MainRow5NullData, PostingAllKeys);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        REPLACE INTO `/Root/main` (`Key`, `Embedding`)
                        VALUES (5u, %s)
                        RETURNING `Data`, `Key`;
                    )sql", EmbOne),
                    MainRow5NullData, PostingAllKeys,
                    TMaybe<TString>(R"([[#;5u]])"));

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        REPLACE INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        SELECT `Key`, %s AS `Embedding`, `Data` FROM `/Root/main`;
                    )sql", EmbOne),
                    MainOriginal, PostingAllKeys);
            }

            Y_UNIT_TEST_TWIN(Insert, Covered) {
                DoTestUpdate(Covered,
                    R"sql(
                        INSERT INTO `/Root/main` (`Key`, `Data`)
                        VALUES (12u, "twelve");
                    )sql",
                    MainWith12, PostingAllKeys);

                DoTestUpdate(Covered,
                    R"sql(
                        INSERT INTO `/Root/main` (`Key`, `Data`)
                        VALUES (12u, "twelve")
                        RETURNING `Data`, `Key`;
                    )sql",
                    MainWith12, PostingAllKeys,
                    TMaybe<TString>(R"([[["twelve"];12u]])"));

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        INSERT INTO `/Root/main` (`Key`, `Embedding`)
                        VALUES (12u, %s);
                    )sql", EmbOne),
                    MainWith12Emb, PostingWith12);

                DoTestUpdate(Covered,
                    Sprintf(R"sql(
                        INSERT INTO `/Root/main` (`Key`, `Embedding`)
                        VALUES (12u, %s)
                        RETURNING `Data`, `Key`;
                    )sql", EmbOne),
                    MainWith12Emb, PostingWith12,
                    TMaybe<TString>(R"([[#;12u]])"));
            }

            Y_UNIT_TEST_TWIN(TruncateTable, Covered) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                UpsertFixture(db);
                AddIndex(db, Covered);

                const TString levelBefore = FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*"));
                const TString codebookBefore = FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*"));
                UNIT_ASSERT_VALUES_EQUAL(ReadIndex(db, "indexImplPostingTable", "*").RowsCount(), 11u);

                {
                    const TString query = R"sql(
                        TRUNCATE TABLE `/Root/main`;
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                CompareYson(R"([])", FormatResultSetYson(ReadMain(db)));
                CompareYson(R"([])", FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "*")));
                CompareYsonUnordered(levelBefore, FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*")));
                CompareYsonUnordered(codebookBefore, FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*")));
            }

            Y_UNIT_TEST_TWIN(TruncateEmptyIndex, Covered) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);
                AddIndex(db, Covered);

                const TString levelBefore = FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*"));
                const TString codebookBefore = FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*"));
                CompareYson(R"([])", FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "*")));

                {
                    const TString query = R"sql(
                        TRUNCATE TABLE `/Root/main`;
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                CompareYson(R"([])", FormatResultSetYson(ReadMain(db)));
                CompareYson(R"([])", FormatResultSetYson(ReadIndex(db, "indexImplPostingTable", "*")));
                CompareYsonUnordered(levelBefore, FormatResultSetYson(ReadIndex(db, "indexImplLevelTable", "*")));
                CompareYsonUnordered(codebookBefore, FormatResultSetYson(ReadIndex(db, "indexImplCodebookTable", "*")));
            }

        } // Y_UNIT_TEST_SUITE(KqpVectorIndexesIvfPq)

    } // namespace NKqp
} // namespace NKikimr
