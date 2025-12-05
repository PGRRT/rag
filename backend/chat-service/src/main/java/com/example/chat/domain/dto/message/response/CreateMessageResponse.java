package com.example.chat.domain.dto.message.response;

import java.util.UUID;

public record CreateMessageResponse(
        UUID id,
        String content
) {
}
