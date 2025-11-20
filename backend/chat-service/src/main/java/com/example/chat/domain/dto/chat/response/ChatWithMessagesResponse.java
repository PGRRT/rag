package com.example.chat.domain.dto.chat.response;

import com.example.chat.domain.dto.message.response.MessageResponse;

import java.util.List;
import java.util.UUID;

public record ChatWithMessagesResponse(
        UUID id,
        String title,
        List<MessageResponse> messages
) {
}
