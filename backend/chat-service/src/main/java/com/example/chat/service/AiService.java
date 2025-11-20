package com.example.chat.service;

import java.util.UUID;

public interface AiService {
    String generateResponse(UUID chatId, String prompt);
    void processAiResponseAsync(UUID chatId, String message);
}
