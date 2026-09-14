#pragma once
#include "../include/adaptq/storage.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

/* -------------------------------------------------------------------------
 * storage/contiguous_slab.h — ContiguousSlabStorage : IStorageBackend
 *
 * Fixed-slot contiguous storage. Ring-buffer eviction. 64-byte-aligned slab.
 * Header-only: all methods are inline so this can be included by both the
 * compiled TU (contiguous_slab.cpp) and test files without ODR violations.
 * ----------------------------------------------------------------------- */

namespace adaptq {

class ContiguousSlabStorage final : public IStorageBackend {
public:
    ContiguousSlabStorage() = default;
    ~ContiguousSlabStorage() override { aligned_free(data_); }

    ContiguousSlabStorage(const ContiguousSlabStorage &)            = delete;
    ContiguousSlabStorage &operator=(const ContiguousSlabStorage &) = delete;

    void init(int capacity, int max_slot_bytes) override {
        assert(capacity > 0 && max_slot_bytes > 0);
        if (data_) aligned_free(data_);
        capacity_   = capacity;
        slot_bytes_ = max_slot_bytes;
        size_ = head_ = next_slot_ = 0;
        k_head_ = k_next_slot_ = k_size_ = 0;
        v_head_ = v_next_slot_ = v_size_ = 0;
        size_t total = (size_t)capacity * (size_t)max_slot_bytes;
        data_ = static_cast<uint8_t *>(aligned_alloc_64(total));
        if (!data_) throw std::bad_alloc();
        memset(data_, 0, total);
        scale_.assign(capacity, 0.f);
        format_tag_.assign(capacity, 0);
    }

    void reset() override {
        size_ = head_ = next_slot_ = 0;
        k_head_ = k_next_slot_ = k_size_ = 0;
        v_head_ = v_next_slot_ = v_size_ = 0;
    }

    StorageSlot write(const uint8_t *data, int data_bytes,
                      float scale, uint8_t format_tag) override {
        assert(data && data_bytes <= slot_bytes_);
        StorageSlot slot;
        if (size_ < capacity_) { slot = next_slot_++; size_++; }
        else                   { slot = head_; head_ = (head_ + 1) % capacity_; }
        write_slot(slot, data, data_bytes, scale, format_tag);
        return slot;
    }

    /* KV-aware regions: the runtime allocates 2 * logical capacity slots,
     * with K in the first half and V in the second half. Each role has its
     * own FIFO ring, so K/V pairs stay associated after wrap-around. */
    StorageSlot write_key(const uint8_t *data, int data_bytes,
                          float scale, uint8_t format_tag) override {
        return write_role(data, data_bytes, scale, format_tag,
                          true, role_capacity());
    }

    StorageSlot write_value(const uint8_t *data, int data_bytes,
                            float scale, uint8_t format_tag) override {
        return write_role(data, data_bytes, scale, format_tag,
                          false, role_capacity());
    }

    CompressResult read(StorageSlot slot) const override {
        assert((int)slot < capacity_);
        return CompressResult{
            data_ + (size_t)slot * slot_bytes_,
            slot_bytes_, scale_[slot], format_tag_[slot], slot
        };
    }

    void free_slot(StorageSlot /*slot*/) override {}  /* ring-buffer: implicit */

    size_t bytes_used()     const override { return (size_t)size_      * slot_bytes_; }
    size_t bytes_capacity() const override { return (size_t)capacity_  * slot_bytes_; }
    const char *name()      const override { return "contiguous_slab"; }

    /* Legacy accessors for KVFlatBuffer-compat code paths */
    int      size()        const { return size_;       }
    int      head_idx()    const { return head_;       }
    int      capacity()    const { return capacity_;   }
    int      slot_bytes()  const { return slot_bytes_; }
    uint8_t *raw_data()          { return data_;       }
    const float *scales()  const { return scale_.data(); }

private:
    int role_capacity() const {
        /* Runtime KV storage is allocated as 2 * logical capacity. */
        return capacity_ / 2;
    }

    StorageSlot write_role(const uint8_t *data, int data_bytes,
                           float scale, uint8_t format_tag,
                           bool is_key, int logical_capacity) {
        assert(logical_capacity > 0);
        int &head = is_key ? k_head_ : v_head_;
        int &next = is_key ? k_next_slot_ : v_next_slot_;
        int &size = is_key ? k_size_ : v_size_;
        const int base = is_key ? 0 : logical_capacity;

        int local;
        if (size < logical_capacity) { local = next++; size++; }
        else { local = head; head = (head + 1) % logical_capacity; }

        StorageSlot slot = (StorageSlot)(base + local);
        write_slot(slot, data, data_bytes, scale, format_tag);
        size_ = k_size_ + v_size_;
        return slot;
    }

    void write_slot(StorageSlot slot, const uint8_t *data, int data_bytes,
                    float scale, uint8_t format_tag) {
        uint8_t *dst = data_ + (size_t)slot * slot_bytes_;
        memcpy(dst, data, data_bytes);
        if (data_bytes < slot_bytes_)
            memset(dst + data_bytes, 0, slot_bytes_ - data_bytes);
        scale_[slot]      = scale;
        format_tag_[slot] = format_tag;
    }

    static uint8_t *aligned_alloc_64(size_t bytes) {
#if defined(_MSC_VER)
        return static_cast<uint8_t *>(_aligned_malloc(bytes, 64));
#else
        void *p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
        return static_cast<uint8_t *>(p);
#endif
    }
    static void aligned_free(void *p) {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    uint8_t             *data_       = nullptr;
    int                  capacity_   = 0;
    int                  slot_bytes_ = 0;
    int                  size_       = 0;
    int                  head_       = 0;
    int                  next_slot_  = 0;
    int                  k_head_     = 0;
    int                  k_next_slot_= 0;
    int                  k_size_    = 0;
    int                  v_head_     = 0;
    int                  v_next_slot_= 0;
    int                  v_size_    = 0;
    std::vector<float>   scale_;
    std::vector<uint8_t> format_tag_;
};

} /* namespace adaptq */
