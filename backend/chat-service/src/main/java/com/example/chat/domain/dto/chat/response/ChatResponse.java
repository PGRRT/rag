package com.example.chat.domain.dto.chat.response;

import com.example.chat.domain.dto.message.response.MessageResponse;

import java.util.List;
import java.util.UUID;

public record ChatResponse(
        UUID id,
        String title
) {
}
