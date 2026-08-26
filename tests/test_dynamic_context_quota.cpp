#include <cassert>
#include <cstdint>
#include <vector>

#include "dynamic_context_quota.hpp"

int main() {
    using llama_box::dynamic_context_quota;

    assert(dynamic_context_quota::for_active_requests(65536, 1) == 65536);
    assert(dynamic_context_quota::for_active_requests(65536, 2) == 32768);
    assert(dynamic_context_quota::for_active_requests(65536, 3) == 21845);
    assert(dynamic_context_quota::position_for_budget(4) == 4);

    assert(dynamic_context_quota::positions_fit({ 12000, 16000 }, 32768));
    assert(!dynamic_context_quota::positions_fit({ 12000, 32769 }, 32768));

    assert(dynamic_context_quota::can_admit(65536, 4, 1, { 12000 }, 32000));
    assert(!dynamic_context_quota::can_admit(65536, 4, 1, { 32769 }, 1000));
    assert(!dynamic_context_quota::can_admit(65536, 4, 1, { 12000 }, 32769));
    assert(!dynamic_context_quota::can_admit(65536, 1, 1, { 1000 }, 1000));
    assert(!dynamic_context_quota::can_admit(65536, 4, 0, {}, 65537));

    assert(dynamic_context_quota::remaining_output_budget(32768, 12000, -1, 0) == 20768);
    assert(dynamic_context_quota::remaining_output_budget(32768, 12000, 10000, 4000) == 6000);
    assert(dynamic_context_quota::remaining_output_budget(32768, 40000, 10000, 0) == 0);

    return 0;
}
