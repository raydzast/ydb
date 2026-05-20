#include <ydb/core/base/ivf_pq.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/strbuf.h>

namespace NKikimr::NIvfPq::NPackedNBitVector {

namespace {

    bool GetGlobalBit(const TStringBuf serialized, const size_t pos) {
        return (static_cast<ui8>(serialized[pos / 8]) >> (pos % 8)) & 1;
    }

    void CheckSerialization(const TVector<ui16>& elements, const size_t nbits, const TStringBuf expected) {
        UNIT_ASSERT_VALUES_EQUAL(Serialize(elements, nbits), expected);
        UNIT_ASSERT_VALUES_EQUAL(Deserialize(expected), elements);
        UNIT_ASSERT_VALUES_EQUAL(expected.size(), CalcByteCount(elements.size(), nbits));
        UNIT_ASSERT_VALUES_EQUAL(CalcElementCount(expected), elements.size());
    }

    void AssertBitAddressingInvariant(const TVector<ui16>& elements, const TStringBuf serialized, const size_t nbits) {
        for (size_t i = 0; i < elements.size(); ++i) {
            for (size_t j = 0; j < nbits; ++j) {
                const size_t pos = i * nbits + j;
                const bool expected = (elements[i] >> j) & 1;
                UNIT_ASSERT_VALUES_EQUAL_C(
                    GetGlobalBit(serialized, pos),
                    expected,
                    "element=" << i << " bit=" << j << " pos=" << pos
                );
            }
        }
    }

} // namespace

Y_UNIT_TEST_SUITE(PackedNBitVector) {

    Y_UNIT_TEST(Serialization) {
        CheckSerialization({}, 4, "\0\x84"_sb);

        CheckSerialization({0}, 1, "\0\x07\x81"_sb);
        CheckSerialization({1}, 1, "\x01\x07\x81"_sb);
        CheckSerialization({0, 1, 0}, 1, "\x02\x05\x81"_sb);

        CheckSerialization({0}, 2, "\0\x06\x82"_sb);
        CheckSerialization({3}, 2, "\x03\x06\x82"_sb);
        CheckSerialization({1, 2, 3}, 2, "\x39\x02\x82"_sb);

        CheckSerialization({5, 3, 7}, 3, "\xDD\x01\x07\x83"_sb);
        CheckSerialization({0, 7}, 3, "\x38\x02\x83"_sb);

        CheckSerialization({0xA, 0x5}, 4, "\x5A\x00\x84"_sb);
        CheckSerialization({0xF}, 4, "\x0F\x04\x84"_sb);
        CheckSerialization({1, 1}, 4, "\x11\x00\x84"_sb);

        CheckSerialization({0, 1}, 8, "\0\1\0\x88"_sb);
        CheckSerialization({255}, 8, "\xFF\x00\x88"_sb);
        CheckSerialization({10, 20, 30}, 8, "\x0A\x14\x1E\x00\x88"_sb);

        CheckSerialization({0x1234, 0x5678}, 16, "\x34\x12\x78\x56\x00\x90"_sb);
        CheckSerialization({0xFFFF}, 16, "\xFF\xFF\x00\x90"_sb);
    }

    Y_UNIT_TEST(BitAddressingInvariant) {
        {
            const TVector<ui16> elements = {5, 3, 7};
            const TString serialized = Serialize(elements, 3);
            AssertBitAddressingInvariant(elements, serialized, 3);
        }
        {
            const TVector<ui16> elements = {1, 2, 3};
            const TString serialized = Serialize(elements, 2);
            AssertBitAddressingInvariant(elements, serialized, 2);
        }
        {
            const TVector<ui16> elements = {0x1234, 0x5678};
            const TString serialized = Serialize(elements, 16);
            AssertBitAddressingInvariant(elements, serialized, 16);
        }
    }

    Y_UNIT_TEST(InvalidInput) {
        UNIT_ASSERT_EXCEPTION(Serialize(TVector<ui16>{0}, 0), yexception);
        UNIT_ASSERT_EXCEPTION(Serialize(TVector<ui16>{0}, 17), yexception);
        UNIT_ASSERT_EXCEPTION(Deserialize("\0\0"_sb), yexception);
        UNIT_ASSERT_EXCEPTION(Deserialize("\0\x10"_sb), yexception);

        const TVector<ui16> tooLarge = {4};
        UNIT_ASSERT_EXCEPTION(Serialize(tooLarge, 2), yexception);
    }

}

} // namespace NKikimr::NIvfPq::NPackedNBitVector
