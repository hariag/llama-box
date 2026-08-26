#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace llama_box {

class dynamic_context_quota {
public:
    static int32_t for_active_requests(int32_t total_context, int32_t active_requests) {
        if (total_context <= 0) {
            return 0;
        }
        return total_context / std::max(1, active_requests);
    }

    static int32_t position_for_budget(int32_t current_position) {
        return std::max<int32_t>(0, current_position);
    }

    static bool positions_fit(const std::vector<int32_t> & positions, int32_t quota) {
        return quota > 0 && std::all_of(positions.begin(), positions.end(), [quota](int32_t position) {
            return position <= quota;
        });
    }

    static bool can_admit(int32_t total_context,
                          int32_t max_parallel,
                          int32_t active_requests,
                          const std::vector<int32_t> & positions,
                          int32_t prompt_tokens) {
        if (total_context <= 0 || max_parallel <= 0 || active_requests < 0 || active_requests >= max_parallel ||
            prompt_tokens < 0) {
            return false;
        }

        const int32_t quota = for_active_requests(total_context, active_requests + 1);
        return positions_fit(positions, quota) && prompt_tokens <= quota;
    }

    static int32_t remaining_output_budget(int32_t context_limit,
                                           int32_t current_position,
                                           int32_t requested_output_tokens,
                                           int32_t generated_tokens) {
        const int64_t context_remaining = std::max<int64_t>(0, int64_t(context_limit) - current_position);
        const int64_t requested_remaining = requested_output_tokens < 0
                                                ? std::numeric_limits<int32_t>::max()
                                                : std::max<int64_t>(0, int64_t(requested_output_tokens) - generated_tokens);
        return int32_t(std::min(context_remaining, requested_remaining));
    }
};

}  // namespace llama_box
